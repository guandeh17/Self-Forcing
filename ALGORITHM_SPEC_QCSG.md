# Query-Conditioned Slot Gating (QCSG) - Cycle 11 Algorithm Specification

**Date**: 2026-04-11  
**Author**: Research Agent  
**Target**: 999-frame video generation (~62.4s @ 16fps)

## Problem Statement

The Cycle 10 float token mechanism has three critical bottlenecks for long-form video generation:

1. **Hard Relevance Filtering**: Uses binary threshold (0.05) to filter float KV slots, causing abrupt inclusion/exclusion of historical context
2. **K-Normalization Loss**: Normalizes K to unit norm then scales to `head_dim^0.5`, destroying magnitude information that helps attention routing
3. **Temporal Decay Asymmetry**: Applies decay only to V values, not to attention logits, creating mismatch where stale slots compete equally in softmax

## Proposed Solution: Query-Conditioned Slot Gating (QCSG)

Replace hard filtering with a soft, query-conditioned gating mechanism that:
- Computes per-slot relevance scores based on query-key similarity
- Uses temperature-scaled softmax for smooth gating weights
- Preserves K magnitude information for attention routing
- Applies unified temporal decay to relevance scores

## Algorithm Specification

### Input
- `query Q`: [B, S_q, H, D] (current generation block queries)
- `float_k`: [K, H, D] (float token keys from all tiers)
- `float_v`: [K, H, D] (float token values)
- `slot_staleness`: [K] (eviction steps since last update)
- `cached_k, cached_v`: [B, S_c, H, D] (local KV cache)

### Output
- `attention_output`: [B, S_q, H, D]

### Steps

#### 1. Compute Query-Slot Relevance Scores
```
For each slot k in [1..K]:
  q_mean = mean(Q, dim=seq)  # [B, H, D]
  relevance[k] = dot(q_mean, float_k[k]) / sqrt(D)  # [B, H]
relevance = mean(relevance, dim=heads)  # [B, K]
```

#### 2. Apply Temporal Decay to Relevance
```
decay_tau = 150  # Increased from 100 for 999-frame generation
temporal_weight[k] = exp(-slot_staleness[k] / decay_tau)
relevance[k] = relevance[k] * temporal_weight[k]
```

#### 3. Compute Soft Gating Weights
```
# Temperature-scaled softmax for smooth gating
gate_weights = softmax(relevance / temperature)  # [B, K], temperature=0.5

# Ensure minimum contribution (prevent complete suppression)
gate_weights = clamp(gate_weights, min=0.01)

# Renormalize after clamping
gate_weights = gate_weights / sum(gate_weights)
```

#### 4. Apply Magnitude-Aware K Scaling
```
# Preserve K magnitude for attention routing
k_norms = norm(float_k, dim=-1)  # [K, H]
target_scale = mean(norm(cached_k, dim=-1))  # scalar
actor = target_scale / (mean(k_norms) + eps)

# Wider clamp than Cycle 10 (0.8-1.2) to allow better magnitude matching
scale_factor = clamp(scale_factor, 0.5, 2.0)

float_k_scaled = float_k * scale_factor  # preserve relative magnitudes
```

#### 5. Gate Float KV Slots
```
# Apply gate weights to V values (not K, to preserve attention routing)
float_v_gated = float_v * gate_weights.unsqueeze(-1).unsqueeze(-1)  # [K, H, D]
```

#### 6. Concatenate and Compute Attention
```
full_k = concat([float_k_scaled, cached_k], dim=seq)
full_v = concat([float_v_gated, cached_v], dim=seq)
output = attention(Q, full_k, full_v)
```

## Key Improvements Over Cycle 10

### 1. Soft Gating via Softmax
- **Before**: Hard threshold (relevance > 0.05) → binary inclusion/exclusion
- **After**: Temperature-scaled softmax → smooth, normalized contribution weights
- **Benefit**: Eliminates abrupt changes in float token contributions, improving temporal consistency

### 2. Magnitude-Aware Scaling
- **Before**: Normalize K to unit norm, then scale to `head_dim^0.5` → loses magnitude information
- **After**: Apply global scale factor while preserving relative magnitudes across slots
- **Benefit**: Attention routing can distinguish important vs. unimportant context based on magnitude

### 3. Unified Temporal Decay
- **Before**: Decay applied only to V values → stale slots still compete equally in softmax
- **After**: Decay applied to relevance scores → affects both attention routing and V contribution
- **Benefit**: Stale slots naturally downweighted in attention computation

### 4. Increased Decay Tau
- **Before**: tau=100 → after 100 evictions (~16s), weight ≈ 0.37
- **After**: tau=150 → after 150 evictions (~24s), weight ≈ 0.37
- **Benefit**: Longer-range context remains influential for 999-frame generation

### 5. Wider Scale Clamp
- **Before**: scale_ratio clamped to [0.8, 1.2] → tight constraint
- **After**: scale_factor clamped to [0.5, 2.0] → allows better magnitude matching
- **Benefit**: Float tokens can be properly scaled relative to cached tokens

## Expected Impact

### Quantitative Predictions
- **Subject Consistency**: +1-2% improvement (0.8938 → 0.90-0.91)
- **Background Consistency**: +1-2% improvement (0.9240 → 0.93-0.94)
- **Temporal Flickering**: Neutral or slight improvement (±0.5%)
- **Motion Smoothness**: Neutral (±0.3%)

### Qualitative Benefits
- Smoother long-range consistency (no abrupt context changes)
- Better preservation of subject identity across 999 frames
- More stable background elements in long videos
- Reduced temporal artifacts from float token injection

## Implementation Details

### File Changes
1. **`wan/modules/float_token_improvements.py`**
   - Added `QueryConditionedSlotGating` class (lines 1777-1960)
   - Implements soft gating, magnitude-aware scaling, and unified temporal decay

2. **`wan/modules/causal_model.py`**
   - Updated import to include `QueryConditionedSlotGating` (line 37)
   - Added `_qcsg_module` initialization in `CausalWanSelfAttention.__init__` (lines 264-267)
   - Replaced hard filtering with QCSG in `forward()` (lines 714-773)

### Hyperparameters
- `temperature`: 0.5 (softmax temperature for gating)
- `decay_tau`: 150.0 (temporal decay time constant)
- `min_gate_weight`: 0.01 (minimum gate weight to prevent complete suppression)
- `scale_clamp`: [0.5, 2.0] (K scaling factor clamp range)

## Validation Plan

### Phase 1: Format Check
- Generate 999-frame video with QCSG
- Verify output format: 999 frames, 480×832 resolution, 16fps
- Check for artifacts: frame drops, corruption, NaN values

### Phase 2: Benchmarking
- Run VBench evaluation on 3 test prompts:
  1. "A golden retriever runs joyfully through a sunlit meadow filled with wildflowers"
  2. "Ocean waves crashing against rocky cliffs at sunset, dramatic clouds overhead"
  3. "A timelapse of clouds moving over mountain peaks, sunlight casting long shadows"
- Compare scores against Cycle 10 baseline (AFT)
- Metrics: subject_consistency, temporal_flickering, motion_smoothness, background_consistency

### Phase 3: Pull Request
- If benchmarks improve, submit PR to https://github.com/MinhoGro/Self-Forcing.git
- Include: algorithm description, benchmark results, generated video samples

## References

### Prior Work
- Cycle 1-10: Iterative improvements to float token mechanism
- Cycle 6: Relevance-based selective float token injection (hard threshold)
- Cycle 7: Lowered relevance threshold from 0.1 to 0.05
- Cycle 10: KV norm quality proxy and momentum-based EMA update

### Key Insights
- Hard thresholding causes abrupt changes in attention context
- K magnitude information is critical for attention routing
- Temporal decay should affect both attention routing and value contribution
- Longer decay tau needed for 999-frame generation (vs. 81-frame benchmarks)

## Conclusion

QCSG addresses the three critical bottlenecks in Cycle 10 by introducing soft gating, magnitude-aware scaling, and unified temporal decay. The algorithm is designed specifically for 999-frame video generation, with increased decay tau and wider scale clamps to handle longer temporal dependencies. Expected improvements: +1-2% on subject/background consistency metrics.
