from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "wan/modules/nki_attention.py"
SPEC = importlib.util.spec_from_file_location("self_forcing_nki_attention_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
NKI_ATTENTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NKI_ATTENTION)
can_use_nki_attention = NKI_ATTENTION.can_use_nki_attention
nki_attention = NKI_ATTENTION.nki_attention


def _torch_attention_cte(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def test_explicit_nki_layout_matches_unmasked_attention() -> None:
    torch.manual_seed(7)
    q = torch.randn(2, 13, 3, 8)
    k = torch.randn(2, 17, 3, 8)
    v = torch.randn(2, 17, 3, 8)

    actual = nki_attention(q, k, v, kernel=_torch_attention_cte)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_explicit_nki_only_pads_query_to_128_and_preserves_exact_kv() -> None:
    seen = []

    def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
        seen.append((tuple(q.shape), tuple(k.shape)))
        return _torch_attention_cte(q, k, v, scale)

    q = torch.randn(1, 13, 3, 8)
    k = torch.randn(1, 17, 3, 8)
    result = nki_attention(q, k, torch.randn_like(k), kernel=kernel)

    assert result.shape == q.shape
    assert seen == [((3, 128, 8), (3, 17, 8))]


def test_explicit_nki_layout_honors_custom_scale() -> None:
    torch.manual_seed(8)
    q = torch.randn(1, 11, 2, 8)
    k = torch.randn(1, 19, 2, 8)
    v = torch.randn(1, 19, 2, 8)
    scale = 0.125

    actual = nki_attention(q, k, v, softmax_scale=scale, kernel=_torch_attention_cte)
    expected = torch.matmul(
        torch.softmax(
            torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * scale,
            dim=-1,
        ),
        v.transpose(1, 2),
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_resident_rolling_cache_shape_selects_explicit_nki(monkeypatch) -> None:
    monkeypatch.setenv("SELF_FORCING_NKI_ATTENTION", "1")
    q = torch.empty(1, 4680, 12, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 6240, 12, 128, dtype=torch.bfloat16)

    assert can_use_nki_attention(q, k, torch.empty_like(k), device_type="neuron")


def test_initial_equal_length_cache_fill_stays_on_sdpa(monkeypatch) -> None:
    monkeypatch.setenv("SELF_FORCING_NKI_ATTENTION", "1")
    q = torch.empty(1, 4680, 12, 128, dtype=torch.bfloat16)

    assert not can_use_nki_attention(q, q, q, device_type="neuron")


def test_nki_path_rejects_unimplemented_semantics(monkeypatch) -> None:
    monkeypatch.setenv("SELF_FORCING_NKI_ATTENTION", "1")
    q = torch.empty(1, 16, 2, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 32, 2, 128, dtype=torch.bfloat16)

    assert not can_use_nki_attention(q, k, k, causal=True, device_type="neuron")
    assert not can_use_nki_attention(q, k, k, dropout_p=0.1, device_type="neuron")
    assert not can_use_nki_attention(q, k, k, q_lens=[16], device_type="neuron")
    assert not can_use_nki_attention(q, k, k, window_size=(8, 0), device_type="neuron")


def test_default_scale_is_inverse_sqrt_head_dim() -> None:
    seen = []

    def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
        seen.append(scale)
        return torch.zeros_like(q)

    q = torch.zeros(1, 4, 2, 64)
    k = torch.zeros(1, 8, 2, 64)
    nki_attention(q, k, torch.zeros_like(k), kernel=kernel)

    assert seen == [1.0 / math.sqrt(64)]
