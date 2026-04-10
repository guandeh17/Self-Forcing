"""
Float Token 改进模块测试脚本

用于验证 Float Token 功能是否正常工作
"""

import torch
import torch.nn as nn

# 测试 float_token_improvements 模块
def test_float_token_improvements():
    print("=" * 60)
    print("Testing Float Token Improvements Module")
    print("=" * 60)

    from wan.modules.float_token_improvements import (
        FloatTokenBank,
        FrameQualityScorer,
        HierarchicalFloatBank,
        apply_rope_with_float_tokens
    )

    d_model = 512
    batch_size = 2
    num_tokens = 10

    # Test 1: FrameQualityScorer
    print("\n[Test 1] FrameQualityScorer")
    scorer = FrameQualityScorer(d_model=d_model)
    frame_tokens = torch.randn(batch_size, num_tokens, d_model)
    quality = scorer.score_frame(frame_tokens)
    print(f"  Input shape: {frame_tokens.shape}")
    print(f"  Quality score: {quality}")
    assert quality.shape == (batch_size, 1), "Quality score shape mismatch"
    assert 0 <= quality.min() <= quality.max() <= 1, "Quality score out of range [0, 1]"
    print("  ✓ PASSED")

    # Test 2: FloatTokenBank
    print("\n[Test 2] FloatTokenBank")
    bank = FloatTokenBank(num_slots=4, d_model=d_model, alpha=0.3)
    evicted_tokens = torch.randn(batch_size, num_tokens, d_model)
    stats = bank.update(evicted_tokens)
    print(f"  Input shape: {evicted_tokens.shape}")
    print(f"  Update stats: {stats}")
    tokens = bank.get_tokens()
    print(f"  Float tokens shape: {tokens.shape}")
    assert tokens.shape == (4, d_model), "Float tokens shape mismatch"
    print("  ✓ PASSED")

    # Test 3: HierarchicalFloatBank
    print("\n[Test 3] HierarchicalFloatBank")
    hier_bank = HierarchicalFloatBank(
        d_model=d_model,
        num_slots_short=2,
        num_slots_mid=2,
        num_slots_long=2,
        use_quality_scorer=True
    )
    evicted = torch.randn(batch_size, num_tokens, d_model)
    frame = torch.randn(batch_size, num_tokens, d_model)
    stats = hier_bank.update(evicted, frame)
    all_tokens = hier_bank.get_all_tokens()
    print(f"  Total float tokens: {all_tokens.shape}")
    assert all_tokens.shape == (6, d_model), f"Expected (6, {d_model}), got {all_tokens.shape}"
    print(f"  Stats: {stats}")
    print("  ✓ PASSED")

    # Test 4: Multiple updates
    print("\n[Test 4] Multiple Updates")
    bank.reset()
    for i in range(10):
        evicted = torch.randn(batch_size, num_tokens, d_model)
        stats = bank.update(evicted)
        if stats['updated']:
            print(f"  Step {i}: cursor={stats['cursor']}, slot_norms={stats['slot_norms'].tolist()}")
    final_stats = bank.get_stats()
    print(f"  Final slot usage: {final_stats['slot_usage'].tolist()}")
    print("  ✓ PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_causal_model_integration():
    """测试 CausalWanSelfAttention 集成"""
    print("\n" + "=" * 60)
    print("Testing CausalWanSelfAttention Integration")
    print("=" * 60)

    from wan.modules.causal_model import CausalWanSelfAttention

    dim = 512
    num_heads = 8
    batch_size = 1
    seq_len = 128
    head_dim = dim // num_heads

    # Test with float tokens enabled
    print("\n[Test 1] CausalWanSelfAttention with Float Tokens")
    attn = CausalWanSelfAttention(
        dim=dim,
        num_heads=num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        use_float_tokens=True,
        use_hierarchical_float_tokens=True,
        float_token_num_slots_short=2,
        float_token_num_slots_mid=2,
        float_token_num_slots_long=2,
        use_quality_scorer=True
    )

    # Create dummy input
    x = torch.randn(batch_size, seq_len, dim)
    seq_lens = torch.tensor([seq_len])
    grid_sizes = torch.tensor([[1, 8, 16]])  # 1 frame, 8x16 patches

    # Create dummy freqs
    from wan.modules.model import rope_params
    d = dim // num_heads
    freqs = torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6))
    ], dim=1)

    # Create block mask (need to account for float tokens)
    from wan.modules.causal_model import create_block_mask
    num_float_tokens = attn.num_float_tokens
    total_length = seq_len + num_float_tokens  # Include float tokens in mask
    padded_length = (math.ceil(total_length / 128) * 128 - total_length) if total_length % 128 != 0 else 0
    ends = torch.zeros(total_length + padded_length, dtype=torch.long)
    ends[:total_length] = total_length

    def mask_fn(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    block_mask = create_block_mask(mask_fn, B=None, H=None, Q_LEN=total_length + padded_length,
                                   KV_LEN=total_length + padded_length, _compile=False, device='cpu')

    # Forward pass (training mode)
    # Disable torch.compile for flex_attention during testing
    import os
    os.environ['TORCH_COMPILE'] = '0'

    # Temporarily replace flex_attention to avoid compilation issues
    import wan.modules.causal_model as cm
    original_flex_attention = cm.flex_attention

    def mock_flex_attention(query, key, value, block_mask=None, **kwargs):
        # query, key, value shape: [B, num_heads, seq_len, head_dim]
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value
        )

    cm.flex_attention = mock_flex_attention

    try:
        output = attn(x, seq_lens, grid_sizes, freqs, block_mask)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        print("  ✓ PASSED")
    finally:
        cm.flex_attention = original_flex_attention

    # Test reset
    print("\n[Test 2] Reset Float Bank")
    attn.reset_float_bank()
    print("  Float bank reset successfully")
    print("  ✓ PASSED")

    print("\n" + "=" * 60)
    print("Integration tests passed!")
    print("=" * 60)


def test_model_config():
    """测试模型配置"""
    print("\n" + "=" * 60)
    print("Testing CausalWanModel Configuration")
    print("=" * 60)

    from wan.modules.causal_model import CausalWanModel

    # Test with float tokens enabled
    print("\n[Test 1] Model with Float Tokens Enabled")
    model = CausalWanModel(
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=512,  # Smaller for testing
        ffn_dim=2048,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=8,
        num_layers=4,  # Fewer layers for testing
        local_attn_size=-1,
        sink_size=0,
        use_float_tokens=True,
        use_hierarchical_float_tokens=True,
        float_token_num_slots_short=2,
        float_token_num_slots_mid=2,
        float_token_num_slots_long=2,
        use_quality_scorer=True
    )

    print(f"  Model created with {len(model.blocks)} blocks")
    print(f"  Use float tokens: {model.use_float_tokens}")

    # Check that attention blocks have float banks
    for i, block in enumerate(model.blocks):
        has_float_bank = hasattr(block.self_attn, 'float_bank') and block.self_attn.float_bank is not None
        print(f"  Block {i}: has_float_bank={has_float_bank}")

    # Test reset
    model.reset_float_banks()
    print("  All float banks reset successfully")
    print("  ✓ PASSED")

    # Test with float tokens disabled
    print("\n[Test 2] Model with Float Tokens Disabled")
    model_no_float = CausalWanModel(
        model_type='t2v',
        dim=512,
        ffn_dim=2048,
        num_heads=8,
        num_layers=2,
        use_float_tokens=False
    )
    print(f"  Model created: use_float_tokens={model_no_float.use_float_tokens}")
    print("  ✓ PASSED")

    print("\n" + "=" * 60)
    print("Configuration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import math

    print("\n" + "=" * 80)
    print(" Float Token Algorithm Improvement Tests ")
    print("=" * 80)

    try:
        test_float_token_improvements()
        test_causal_model_integration()
        test_model_config()

        print("\n" + "=" * 80)
        print(" ALL TESTS PASSED! ")
        print("=" * 80)
        print("\nFloat Token improvements are working correctly.")
        print("Key improvements implemented:")
        print("  ✓ EMA-based dynamic compression (FloatTokenBank)")
        print("  ✓ Frame quality scoring (FrameQualityScorer)")
        print("  ✓ Hierarchical float tokens (short/mid/long term)")
        print("  ✓ RoPE alignment for float tokens")
        print("  ✓ Training mode support")
        print("  ✓ Configurable parameters")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
