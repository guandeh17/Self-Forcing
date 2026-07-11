"""Optional AWS NKI attention bridge for Self-Forcing on Trainium.

Torch-Neuron's automatic NKI SDPA lowering requires both sequence dimensions
to be divisible by 512. Self-Forcing's resident inference shape is 4,680 query
tokens over a 6,240-token rolling KV cache, so this explicit bridge pads only
the throwaway query rows required by ``attention_cte``.

All Neuron imports stay lazy so CUDA and CPU users need no AWS dependencies.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import torch

_NKI_ATTENTION_OP: Callable[..., torch.Tensor] | None = None
_NKI_ATTENTION_ERROR: BaseException | None = None
_NKI_QUERY_TILE = 128


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def nki_attention_enabled() -> bool:
    return _env_flag("SELF_FORCING_NKI_ATTENTION", "0")


def can_use_nki_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens: Any = None,
    k_lens: Any = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    device_type: str | None = None,
) -> bool:
    """Return whether this call matches the proven ``attention_cte`` contract."""

    if not nki_attention_enabled():
        return False
    if (device_type or q.device.type) != "neuron":
        return False
    if q_lens is not None or k_lens is not None:
        return False
    if dropout_p != 0.0 or causal or window_size != (-1, -1):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        return False
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        return False
    # NeuronXCC currently crashes on Self-Forcing's initial Q=KV=4,680
    # specialization. Its resident Q=4,680, KV=6,240 specialization is proven.
    if k.shape[1] <= q.shape[1]:
        return False
    if q.shape[-1] not in (64, 128) or q.shape[0] * q.shape[2] > 512:
        return False
    return q.dtype in (torch.bfloat16, torch.float16) and k.dtype == q.dtype == v.dtype


def _load_nki_attention_op() -> Callable[..., torch.Tensor]:
    global _NKI_ATTENTION_ERROR, _NKI_ATTENTION_OP
    if _NKI_ATTENTION_OP is not None:
        return _NKI_ATTENTION_OP
    if _NKI_ATTENTION_ERROR is not None:
        raise RuntimeError("Self-Forcing NKI attention initialization previously failed") from (
            _NKI_ATTENTION_ERROR
        )

    try:
        from nkilib.core.attention.attention_cte import attention_cte
        from torch_neuronx import nki_op, wrap_nki
        from torch_neuronx.utils import get_logical_neuron_cores

        wrapped_attention = wrap_nki(attention_cte)

        @nki_op("self_forcing::attention_cte_v1", mutates_args={})
        def attention_cte_op(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            scale: float,
        ) -> torch.Tensor:
            grid = (int(get_logical_neuron_cores()),)
            return wrapped_attention[grid](
                q,
                k,
                v,
                tp_q=True,
                tp_k=True,
                scale=scale,
                causal_mask=False,
                cache_softmax=False,
            )

        _NKI_ATTENTION_OP = attention_cte_op
        return attention_cte_op
    except BaseException as exc:
        _NKI_ATTENTION_ERROR = exc
        raise


def initialize_nki_attention() -> None:
    """Register the custom operator before Dynamo traces model execution."""

    if nki_attention_enabled():
        _load_nki_attention_op()


@torch.compiler.disable
def _invoke_nki_attention_isolated(
    op: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return op(q, k, v, scale)


def nki_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run ``[B, S, H, D]`` unmasked attention through NKI ``attention_cte``."""

    batch, q_len, heads, head_dim = q.shape
    k_len = k.shape[1]
    scale = float(softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim))

    q_nki = q.permute(0, 2, 1, 3).contiguous().reshape(batch * heads, q_len, head_dim)
    k_nki = k.permute(0, 2, 1, 3).contiguous().reshape(batch * heads, k_len, head_dim)
    v_nki = v.permute(0, 2, 1, 3).contiguous().reshape(batch * heads, k_len, head_dim)

    padded_q_len = math.ceil(q_len / _NKI_QUERY_TILE) * _NKI_QUERY_TILE
    if padded_q_len != q_len:
        q_nki = torch.cat(
            [q_nki, q_nki.new_zeros((batch * heads, padded_q_len - q_len, head_dim))],
            dim=1,
        )

    op = kernel or _load_nki_attention_op()
    if kernel is None and _env_flag("SELF_FORCING_NKI_ISOLATE_OP", "1"):
        out = _invoke_nki_attention_isolated(op, q_nki, k_nki, v_nki, scale)
    else:
        out = op(q_nki, k_nki, v_nki, scale)
    out = out[:, :q_len].to(q.dtype)
    return out.reshape(batch, heads, q_len, head_dim).permute(0, 2, 1, 3).contiguous()


@torch.compiler.disable
def nki_attention_isolated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Keep the complete layout bridge outside the surrounding Dynamo graph."""

    return nki_attention(q, k, v, softmax_scale=softmax_scale)


def nki_attention_status() -> dict[str, Any]:
    return {
        "enabled": nki_attention_enabled(),
        "initialized": _NKI_ATTENTION_OP is not None,
        "initialization_error": repr(_NKI_ATTENTION_ERROR) if _NKI_ATTENTION_ERROR else None,
        "kernel": "nkilib.core.attention.attention_cte",
        "requires_cache_expansion": True,
        "query_tile": _NKI_QUERY_TILE,
    }
