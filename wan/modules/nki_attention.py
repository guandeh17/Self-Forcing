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
_NKI_ATTENTION_OUTPUT_PROJECTION_OP: Callable[..., torch.Tensor] | None = None
_NKI_ATTENTION_OUTPUT_PROJECTION_ERROR: BaseException | None = None
_NKI_PROJECTION_WEIGHT_CACHE: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
_NKI_QUERY_TILE = 128


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def nki_attention_enabled() -> bool:
    return _env_flag("SELF_FORCING_NKI_ATTENTION", "0")


def nki_attention_output_projection_enabled() -> bool:
    return nki_attention_enabled() and _env_flag(
        "SELF_FORCING_NKI_ATTENTION_OUTPUT_PROJECTION", "1"
    )


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
        raise RuntimeError(
            "Self-Forcing NKI attention initialization previously failed"
        ) from (_NKI_ATTENTION_ERROR)

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
    except Exception as exc:
        _NKI_ATTENTION_ERROR = exc
        raise


def _load_nki_attention_output_projection_op() -> Callable[..., torch.Tensor]:
    global _NKI_ATTENTION_OUTPUT_PROJECTION_ERROR, _NKI_ATTENTION_OUTPUT_PROJECTION_OP
    if _NKI_ATTENTION_OUTPUT_PROJECTION_OP is not None:
        return _NKI_ATTENTION_OUTPUT_PROJECTION_OP
    if _NKI_ATTENTION_OUTPUT_PROJECTION_ERROR is not None:
        raise RuntimeError(
            "Self-Forcing fused NKI attention/output projection initialization previously failed"
        ) from _NKI_ATTENTION_OUTPUT_PROJECTION_ERROR

    try:
        import nki
        from nkilib.core.attention.attention_cte import attention_cte
        from nkilib.core.output_projection.output_projection_cte import (
            output_projection_cte,
        )
        from torch_neuronx import nki_op, wrap_nki
        from torch_neuronx.utils import get_logical_neuron_cores

        @nki.jit
        def attention_output_projection_kernel(q, k, v, weight, bias, scale):
            attention = attention_cte(
                q,
                k,
                v,
                scale=scale,
                causal_mask=False,
                tp_q=True,
                tp_k=True,
                tp_out=True,
                cache_softmax=False,
            )
            return output_projection_cte(
                attention.reshape((1, q.shape[0], q.shape[2], q.shape[1])),
                weight,
                bias,
            )

        wrapped = wrap_nki(attention_output_projection_kernel)

        @nki_op("self_forcing::attention_output_projection_cte_v1", mutates_args={})
        def attention_output_projection_op(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            scale: float,
        ) -> torch.Tensor:
            grid = (int(get_logical_neuron_cores()),)
            return wrapped[grid](q, k, v, weight, bias, scale)

        _NKI_ATTENTION_OUTPUT_PROJECTION_OP = attention_output_projection_op
        return attention_output_projection_op
    except Exception as exc:
        _NKI_ATTENTION_OUTPUT_PROJECTION_ERROR = exc
        raise


def initialize_nki_attention() -> None:
    """Register the custom operator before Dynamo traces model execution."""

    if nki_attention_enabled():
        _load_nki_attention_op()
        if nki_attention_output_projection_enabled():
            _load_nki_attention_output_projection_op()


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
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]
    | None = None,
) -> torch.Tensor:
    """Run ``[B, S, H, D]`` unmasked attention through NKI ``attention_cte``."""

    batch, q_len, heads, head_dim = q.shape
    k_len = k.shape[1]
    scale = float(
        softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    )

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


def can_use_nki_attention_output_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    device_type: str | None = None,
) -> bool:
    """Return whether the fused resident attention/projection kernel is valid."""

    if not nki_attention_output_projection_enabled():
        return False
    if not can_use_nki_attention(q, k, v, device_type=device_type):
        return False
    if q.shape[0] != 1 or q.shape[-1] != 128:
        return False
    hidden = q.shape[2] * q.shape[3]
    return (
        weight.ndim == 2
        and tuple(weight.shape) == (hidden, hidden)
        and weight.dtype == q.dtype
        and bias is not None
        and bias.ndim == 1
        and bias.shape[0] == hidden
        and bias.dtype == q.dtype
    )


def _projection_weight_for_nki(weight: torch.Tensor) -> torch.Tensor:
    """Cache Linear's [out, in] weight once in NKI's [in, out] layout."""

    cache_key = id(weight)
    version = int(weight._version)
    cached = _NKI_PROJECTION_WEIGHT_CACHE.get(cache_key)
    if cached is not None and cached[0] is weight and cached[1] == version:
        return cached[2]
    transposed = weight.detach().transpose(0, 1).contiguous()
    _NKI_PROJECTION_WEIGHT_CACHE[cache_key] = (weight, version, transposed)
    return transposed


@torch.compiler.disable
def nki_attention_output_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    kernel: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Fuse resident attention and its Linear output projection in one NKI op."""

    batch, q_len, heads, head_dim = q.shape
    scale = float(
        softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
    )
    q_nki = q.permute(0, 2, 1, 3).contiguous().reshape(heads, q_len, head_dim)
    k_nki = k.permute(0, 2, 1, 3).contiguous().reshape(heads, k.shape[1], head_dim)
    v_nki = v.permute(0, 2, 1, 3).contiguous().reshape(heads, v.shape[1], head_dim)

    padded_q_len = math.ceil(q_len / _NKI_QUERY_TILE) * _NKI_QUERY_TILE
    if padded_q_len != q_len:
        q_nki = torch.cat(
            [q_nki, q_nki.new_zeros((heads, padded_q_len - q_len, head_dim))], dim=1
        )

    op = kernel or _load_nki_attention_output_projection_op()
    out = op(
        q_nki,
        k_nki,
        v_nki,
        _projection_weight_for_nki(weight),
        bias.reshape(1, -1).contiguous(),
        scale,
    )
    return out[:, :q_len].to(q.dtype).reshape(batch, q_len, heads * head_dim)


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
        "initialization_error": repr(_NKI_ATTENTION_ERROR)
        if _NKI_ATTENTION_ERROR
        else None,
        "kernel": "nkilib.core.attention.attention_cte",
        "output_projection_enabled": nki_attention_output_projection_enabled(),
        "output_projection_initialized": _NKI_ATTENTION_OUTPUT_PROJECTION_OP
        is not None,
        "output_projection_initialization_error": repr(
            _NKI_ATTENTION_OUTPUT_PROJECTION_ERROR
        )
        if _NKI_ATTENTION_OUTPUT_PROJECTION_ERROR
        else None,
        "output_projection_kernel": "attention_cte(tp_out=True) + output_projection_cte",
        "requires_cache_expansion": True,
        "query_tile": _NKI_QUERY_TILE,
    }
