# Float Token Improvements for 1-Minute Video Generation - Implementation Specification

**Date:** 2026-04-11  
**Target:** 1-minute (1920 frame @ 16fps) video generation  
**Priority:** Long-term temporal consistency improvement

## Overview

Implement enhanced float token mechanisms specifically optimized for 1-minute video generation. The current implementation uses static configurations that are suboptimal for 1920-frame sequences.

## Key Improvements

### 1. Dynamic Update Intervals Based on Content Dynamics
**Problem:** Current fixed intervals (1, 30, 90) don't adapt to video content.  
**Solution:** Implement content-aware interval adjustment using the existing content-adaptive alpha as a proxy for scene dynamics.

### 2. Layer-Adaptive Float Token Configuration
**Problem:** All 32 transformer layers use identical float token configuration, wasting memory.  
**Solution:** Configure different slot counts based on layer depth (early/mid/late).

### 3. Temporal Coherence Scoring
**Problem:** Current FrameQualityScorer only looks at consecutive frames.  
**Solution:** Implement multi-frame temporal coherence scoring.

### 4. Progressive Bank Activation
**Problem:** Long-term bank activates immediately, causing early-frame bias.  
**Solution:** Gradually increase long-term bank influence as video progresses.

## Implementation Tasks

### Task 1: Add DynamicIntervalScheduler to float_token_improvements.py

```python
class DynamicIntervalScheduler:
    """
    Adjust update intervals based on content stability.
    
    High similarity (stable scene) -> longer intervals
    Low similarity (scene change) -> shorter intervals
    """
    
    def __init__(self, base_interval: int, min_factor: float = 0.5, 
                 max_factor: float = 3.0, history_size: int = 10):
        self.base_interval = base_interval
        self.min_interval = max(1, int(base_interval * min_factor))
        self.max_interval = int(base_interval * max_factor)
        self.stability_history = deque(maxlen=history_size)
    
    def get_interval(self, cosine_similarity: float) -> int:
        """Compute dynamic interval based on stability."""
        stability_score = (cosine_similarity + 1) / 2  # Normalize to [0, 1]
        self.stability_history.append(stability_score)
        avg_stability = sum(self.stability_history) / len(self.stability_history)
        
        # More stable = longer interval
        interval = int(self.base_interval * (1 + 2 * avg_stability))
        return max(self.min_interval, min(self.max_interval, interval))
```

**Requirements:**
- Add to `float_token_improvements.py`
- Integrate into `FloatKVSlot` class
- Add `use_dynamic_intervals` parameter (default: False)

### Task 2: Add Layer-Adaptive Configuration

```python
# Layer configuration presets for Wan 1.3B (32 layers)
LAYER_CONFIGS = {
    'memory_efficient': {
        'early': {'layers': range(0, 10), 'short': 2, 'mid': 0, 'long': 0},
        'middle': {'layers': range(10, 22), 'short': 4, 'mid': 4, 'long': 2},
        'late': {'layers': range(22, 32), 'short': 4, 'mid': 4, 'long': 8},
    },
    'uniform': {
        'all': {'short': 4, 'mid': 4, 'long': 4}
    }
}

def get_layer_float_config(layer_idx: int, num_layers: int = 32, 
                           preset: str = 'memory_efficient') -> dict:
    """Get float token config for specific layer."""
    if preset == 'uniform':
        return LAYER_CONFIGS['uniform']['all']
    
    for tier, config in LAYER_CONFIGS['memory_efficient'].items():
        if layer_idx in config['layers']:
            return {k: v for k, v in config.items() if k != 'layers'}
    
    # Default to middle config
    return {k: v for k, v in LAYER_CONFIGS['memory_efficient']['middle'].items() 
            if k != 'layers'}
```

**Requirements:**
- Add configuration presets to `float_token_improvements.py`
- Modify `CausalWanModel.__init__` to accept `use_layer_adaptive_float_tokens` parameter
- Pass layer index to each `CausalWanAttentionBlock`
- Configure float banks per-layer based on preset

### Task 3: Add TemporalCoherenceScorer

```python
class TemporalCoherenceScorer(nn.Module):
    """
    Multi-frame temporal coherence scoring.
    
    Tracks consistency across multiple timescales:
    - Short (2-3 frames): Detect flickering
    - Mid (5-10 frames): Detect gradual drift
    - Long (20+ frames): Detect semantic inconsistency
    """
    
    def __init__(self, d_model: int, history_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.history_size = history_size
        
        # Frame history buffer
        self.register_buffer('frame_history', torch.zeros(history_size, d_model))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('history_count', torch.tensor(0, dtype=torch.long))
    
    def update_history(self, frame_vec: torch.Tensor):
        """Add frame to history buffer."""
        idx = self.history_ptr.item()
        self.frame_history[idx] = frame_vec.detach().mean(dim=0)
        self.history_ptr = (self.history_ptr + 1) % self.history_size
        self.history_count = min(self.history_count + 1, self.history_size)
    
    def compute_coherence(self, current_frame: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute temporal coherence scores.
        
        Returns:
            scores: [short, mid, long] coherence scores
            overall: weighted overall coherence
        """
        if self.history_count < 5:
            return torch.ones(3), 1.0
        
        current_vec = current_frame.mean(dim=0)
        
        # Get history windows
        short_window = self._get_recent_frames(3)
        mid_window = self._get_recent_frames(10)
        long_window = self._get_recent_frames(min(30, self.history_count.item()))
        
        # Compute similarities
        short_sim = F.cosine_similarity(current_vec, short_window.mean(dim=0), dim=0)
        mid_sim = F.cosine_similarity(current_vec, mid_window.mean(dim=0), dim=0)
        long_sim = F.cosine_similarity(current_vec, long_window.mean(dim=0), dim=0)
        
        scores = torch.stack([short_sim, mid_sim, long_sim])
        overall = scores.prod().item() ** (1/3)  # Geometric mean
        
        return scores, overall
```

**Requirements:**
- Add to `float_token_improvements.py`
- Modify `HierarchicalFloatKVBank` to optionally use coherence scorer
- Replace simple quality scoring when `use_temporal_coherence=True`

### Task 4: Add Progressive Bank Activation

```python
class ProgressiveBankActivation:
    """
    Gradually activate long-term float tokens as video progresses.
    Prevents early-frame bias in long-term memory.
    """
    
    def __init__(self, warmup_frames: int = 300):  # ~20 seconds at 16fps
        self.warmup_frames = warmup_frames
        self.frame_count = 0
    
    def get_long_term_weight(self) -> float:
        """Get weight for long-term bank [0, 1]."""
        if self.frame_count < self.warmup_frames:
            return self.frame_count / self.warmup_frames
        return 1.0
    
    def get_effective_alpha(self, base_alpha: float, bank_type: str = 'short') -> float:
        """Adjust alpha based on progression and bank type."""
        if bank_type == 'short':
            return base_alpha
        elif bank_type == 'mid':
            progress = min(1.0, self.frame_count / (self.warmup_frames / 2))
            return base_alpha * (1 + 0.3 * progress)
        else:  # long
            weight = self.get_long_term_weight()
            return base_alpha * (0.3 + 0.7 * weight)
    
    def step(self, num_frames: int = 1):
        """Advance frame counter."""
        self.frame_count += num_frames
```

**Requirements:**
- Add to `float_token_improvements.py`
- Integrate into `HierarchicalFloatKVBank`
- Call `step()` after each frame update
- Use `get_effective_alpha()` in `FloatKVSlot.update()`

### Task 5: Update CausalWanModel Integration

**Modify `CausalWanModel.__init__`:**
```python
def __init__(self, ...,
             use_layer_adaptive_float_tokens: bool = False,
             layer_config_preset: str = 'memory_efficient',
             use_dynamic_intervals: bool = False,
             use_temporal_coherence: bool = False,
             use_progressive_activation: bool = False,
             progressive_warmup_frames: int = 300,
             ...):
```

**Modify block creation:**
```python
for layer_idx in range(num_layers):
    # Get layer-specific config if adaptive
    if use_layer_adaptive_float_tokens:
        layer_config = get_layer_float_config(layer_idx, num_layers, layer_config_preset)
        # Override slot counts for this layer
        layer_short = layer_config['short']
        layer_mid = layer_config['mid']
        layer_long = layer_config['long']
    else:
        layer_short = float_token_num_slots_short
        layer_mid = float_token_num_slots_mid
        layer_long = float_token_num_slots_long
    
    self.blocks.append(CausalWanAttentionBlock(
        ...,
        layer_idx=layer_idx,  # Pass layer index
        float_token_num_slots_short=layer_short,
        float_token_num_slots_mid=layer_mid,
        float_token_num_slots_long=layer_long,
        use_dynamic_intervals=use_dynamic_intervals,
        use_temporal_coherence=use_temporal_coherence,
        use_progressive_activation=use_progressive_activation,
        progressive_warmup_frames=progressive_warmup_frames,
        ...
    ))
```

### Task 6: Add Configuration Helper

```python
def create_adaptive_float_token_config(
    num_target_frames: int = 1920,  # 1 minute @ 16fps
    use_layer_adaptive: bool = True,
    use_dynamic_intervals: bool = True,
    use_temporal_coherence: bool = True,
    use_progressive_activation: bool = True,
) -> dict:
    """
    Create optimal float token configuration for long video generation.
    
    Returns model_kwargs dict with all adaptive features enabled.
    """
    return {
        "use_float_tokens": True,
        "use_kv_bank_v2": True,
        "use_layer_adaptive_float_tokens": use_layer_adaptive,
        "layer_config_preset": "memory_efficient",
        "use_dynamic_intervals": use_dynamic_intervals,
        "dynamic_interval_min_factor": 0.5,
        "dynamic_interval_max_factor": 3.0,
        "use_temporal_coherence": use_temporal_coherence,
        "coherence_history_size": 30,
        "use_progressive_activation": use_progressive_activation,
        "progressive_warmup_frames": min(300, num_target_frames // 6),
        # Base slot counts (will be overridden by layer-adaptive config)
        "float_token_num_slots_short": 4,
        "float_token_num_slots_mid": 4,
        "float_token_num_slots_long": 4,
    }
```

## Files to Modify

1. `/workspace/Self-Forcing/wan/modules/float_token_improvements.py`
   - Add `DynamicIntervalScheduler`
   - Add `TemporalCoherenceScorer`
   - Add `ProgressiveBankActivation`
   - Add `get_layer_float_config()` helper
   - Add `create_adaptive_float_token_config()` helper
   - Modify `FloatKVSlot` for dynamic intervals
   - Modify `HierarchicalFloatKVBank` for coherence and progressive activation

2. `/workspace/Self-Forcing/wan/modules/causal_model.py`
   - Modify `CausalWanSelfAttention.__init__` for new parameters
   - Modify `CausalWanAttentionBlock.__init__` for layer index
   - Modify `CausalWanModel.__init__` for layer-adaptive configuration

## Testing Requirements

1. Unit test `DynamicIntervalScheduler` with high/low similarity
2. Unit test `ProgressiveBankActivation` warmup
3. Integration test: Create model with all features enabled
4. Generate 1-minute video and verify no errors
5. Compare memory usage vs baseline

## Compatibility

- All new parameters have defaults preserving backward compatibility
- Existing configs work without modification
- Features are opt-in via boolean flags
