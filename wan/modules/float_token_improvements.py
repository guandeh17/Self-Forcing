"""
Float Token 算法改进模块

包含改进的 Float Token 实现：
- FloatTokenBank: EMA 动态更新机制
- FrameQualityScorer: 帧质量感知筛选
- HierarchicalFloatBank: 分层 Float Token 设计
- DynamicIntervalScheduler: 基于内容稳定性的动态更新间隔
- TemporalCoherenceScorer: 多帧时间一致性评分
- ProgressiveBankActivation: 渐进式银行激活
- RoPE 对齐辅助函数

作者: Claude
日期: 2026-04-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple
from collections import deque


# =============================================================================
# Layer-Adaptive Configuration Helpers
# =============================================================================

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
    """
    Get float token config for specific layer.
    
    Args:
        layer_idx: Layer index (0 to num_layers-1)
        num_layers: Total number of layers
        preset: Configuration preset ('memory_efficient' or 'uniform')
        
    Returns:
        Dict with 'short', 'mid', 'long' slot counts
    """
    if preset == 'uniform':
        return LAYER_CONFIGS['uniform']['all'].copy()
    
    for tier, config in LAYER_CONFIGS['memory_efficient'].items():
        if layer_idx in config['layers']:
            return {k: v for k, v in config.items() if k != 'layers'}
    
    # Default to middle config
    return {k: v for k, v in LAYER_CONFIGS['memory_efficient']['middle'].items()
            if k != 'layers'}


def create_adaptive_float_token_config(
    num_target_frames: int = 1920,  # 1 minute @ 16fps
    use_layer_adaptive: bool = True,
    use_dynamic_intervals: bool = True,
    use_temporal_coherence: bool = True,
    use_progressive_activation: bool = True,
) -> dict:
    """
    Create optimal float token configuration for long video generation.
    
    Args:
        num_target_frames: Target number of frames for video generation
        use_layer_adaptive: Enable layer-adaptive slot configuration
        use_dynamic_intervals: Enable dynamic update intervals
        use_temporal_coherence: Enable temporal coherence scoring
        use_progressive_activation: Enable progressive bank activation
        
    Returns:
        model_kwargs dict with all adaptive features enabled
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


# =============================================================================
# Dynamic Update Interval Scheduler
# =============================================================================

class DynamicIntervalScheduler:
    """
    Adjust update intervals based on content stability.
    
    High similarity (stable scene) -> longer intervals
    Low similarity (changing scene) -> shorter intervals
    
    Args:
        base_interval: Base update interval
        min_factor: Minimum interval factor (default 0.5)
        max_factor: Maximum interval factor (default 3.0)
        history_size: Size of stability history for smoothing
    """
    
    def __init__(self, base_interval: int, min_factor: float = 0.5,
                 max_factor: float = 3.0, history_size: int = 10):
        self.base_interval = base_interval
        self.min_interval = max(1, int(base_interval * min_factor))
        self.max_interval = int(base_interval * max_factor)
        self.stability_history = deque(maxlen=history_size)
    
    def get_interval(self, cosine_similarity: float) -> int:
        """
        Compute dynamic interval based on stability.
        
        Args:
            cosine_similarity: Cosine similarity between current and previous content
                               Range: [-1, 1]
        
        Returns:
            Dynamic interval (integer)
        """
        # Normalize to [0, 1]
        stability_score = (cosine_similarity + 1) / 2
        self.stability_history.append(stability_score)
        avg_stability = sum(self.stability_history) / len(self.stability_history)
        
        # More stable = longer interval (up to 3x base)
        interval = int(self.base_interval * (1 + 2 * avg_stability))
        return max(self.min_interval, min(self.max_interval, interval))
    
    def reset(self):
        """Reset the scheduler state."""
        self.stability_history.clear()


# =============================================================================
# Temporal Coherence Scorer
# =============================================================================

class TemporalCoherenceScorer(nn.Module):
    """
    Multi-frame temporal coherence scoring.
    
    Tracks consistency across multiple timescales:
    - Short (2-3 frames): Detect flickering
    - Mid (5-10 frames): Detect gradual drift
    - Long (20+ frames): Detect semantic inconsistency
    
    Args:
        d_model: Model dimension
        history_size: Number of frames to keep in history
    """
    
    def __init__(self, d_model: int, history_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.history_size = history_size
        
        # Frame history buffer
        self.register_buffer('frame_history', torch.zeros(history_size, d_model))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('history_count', torch.tensor(0, dtype=torch.long))
    
    def reset(self):
        """Reset the scorer state."""
        self.frame_history.zero_()
        self.history_ptr.zero_()
        self.history_count.zero_()
    
    def update_history(self, frame_vec: torch.Tensor):
        """
        Add frame to history buffer.
        
        Args:
            frame_vec: Frame vector [d_model] or [seq_len, d_model]
        """
        with torch.no_grad():
            idx = self.history_ptr.item()
            if frame_vec.dim() == 2:
                frame_vec = frame_vec.mean(dim=0)
            self.frame_history[idx] = frame_vec.detach()
            self.history_ptr = (self.history_ptr + 1) % self.history_size
            self.history_count = min(self.history_count + 1, self.history_size)
    
    def _get_recent_frames(self, n: int) -> torch.Tensor:
        """Get the most recent n frames from history."""
        count = self.history_count.item()
        n = min(n, count)
        
        if n == 0:
            return self.frame_history[:1]
        
        end_idx = self.history_ptr.item()
        start_idx = (end_idx - n) % self.history_size
        
        if start_idx < end_idx:
            return self.frame_history[start_idx:end_idx]
        else:
            # Wrap around
            return torch.cat([self.frame_history[start_idx:], self.frame_history[:end_idx]], dim=0)
    
    def compute_coherence(self, current_frame: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute temporal coherence scores.
        
        Args:
            current_frame: Current frame tokens [seq_len, d_model] or [d_model]
        
        Returns:
            scores: [short, mid, long] coherence scores
            overall: weighted overall coherence (geometric mean)
        """
        if self.history_count < 5:
            return torch.ones(3, device=current_frame.device), 1.0
        
        with torch.no_grad():
            if current_frame.dim() == 2:
                current_vec = current_frame.mean(dim=0)
            else:
                current_vec = current_frame
            
            # Get history windows
            short_window = self._get_recent_frames(3)
            mid_window = self._get_recent_frames(10)
            long_window = self._get_recent_frames(min(30, self.history_count.item()))
            
            # Compute similarities
            short_sim = F.cosine_similarity(
                current_vec.unsqueeze(0),
                short_window.mean(dim=0, keepdim=True),
                dim=-1
            ).mean()
            
            mid_sim = F.cosine_similarity(
                current_vec.unsqueeze(0),
                mid_window.mean(dim=0, keepdim=True),
                dim=-1
            ).mean()
            
            long_sim = F.cosine_similarity(
                current_vec.unsqueeze(0),
                long_window.mean(dim=0, keepdim=True),
                dim=-1
            ).mean()
            
            scores = torch.stack([short_sim, mid_sim, long_sim])
            # Normalize from [-1, 1] to [0, 1] for geometric mean
            scores_normalized = (scores + 1) / 2
            # Geometric mean for overall coherence
            overall = (scores_normalized.prod().item() + 1e-8) ** (1/3)
            
            return scores, overall
    
    def forward(self, current_frame: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Forward interface, same as compute_coherence."""
        return self.compute_coherence(current_frame)


# =============================================================================
# Progressive Bank Activation
# =============================================================================

class ProgressiveBankActivation:
    """
    Gradually activate long-term float tokens as video progresses.
    Prevents early-frame bias in long-term memory.
    
    Args:
        warmup_frames: Number of frames for warmup (default 300 ~ 20s @ 16fps)
    """
    
    def __init__(self, warmup_frames: int = 300):
        self.warmup_frames = warmup_frames
        self.frame_count = 0
    
    def get_long_term_weight(self) -> float:
        """
        Get weight for long-term bank [0, 1].
        
        Returns:
            Weight value between 0 and 1
        """
        if self.frame_count < self.warmup_frames:
            return self.frame_count / self.warmup_frames
        return 1.0
    
    def get_effective_alpha(self, base_alpha: float, bank_type: str = 'short') -> float:
        """
        Adjust alpha based on progression and bank type.
        
        Args:
            base_alpha: Base alpha value
            bank_type: Type of bank ('short', 'mid', or 'long')
        
        Returns:
            Adjusted alpha value
        """
        if bank_type == 'short':
            return base_alpha
        elif bank_type == 'mid':
            progress = min(1.0, self.frame_count / (self.warmup_frames / 2))
            return base_alpha * (1 + 0.3 * progress)
        else:  # long
            weight = self.get_long_term_weight()
            return base_alpha * (0.3 + 0.7 * weight)
    
    def step(self, num_frames: int = 1):
        """
        Advance frame counter.
        
        Args:
            num_frames: Number of frames to advance
        """
        self.frame_count += num_frames
    
    def reset(self):
        """Reset the activation state."""
        self.frame_count = 0
    
    def get_progress(self) -> float:
        """
        Get current progress as a ratio [0, 1].
        
        Returns:
            Progress ratio
        """
        return min(1.0, self.frame_count / self.warmup_frames)


# =============================================================================
# Original Components
# =============================================================================


class FrameQualityScorer(nn.Module):
    """
    帧质量评分模块

    用于评估单帧的质量，筛选出高质量帧用于更新 Float Tokens。
    检测指标：
    - 帧间相似度（检测突变/崩坏）
    - 帧内方差（检测模糊/噪声）
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.prev_frame_vec = None

        # 质量评分网络
        self.similarity_proj = nn.Linear(d_model, d_model // 4)
        self.variance_proj = nn.Linear(d_model, d_model // 4)
        self.quality_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def reset(self):
        """重置状态，用于新序列开始时"""
        self.prev_frame_vec = None

    def score_frame(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """
        计算帧质量分数

        Args:
            frame_tokens: 单帧的 tokens，shape [B, seq_len, d_model] 或 [seq_len, d_model]

        Returns:
            quality_score: 质量分数 [0, 1]，shape [B] 或 scalar
        """
        # 确保是 3D
        if frame_tokens.dim() == 2:
            frame_tokens = frame_tokens.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = frame_tokens.shape[0]

        # 1. 帧池化 - 将帧内所有 tokens 平均
        frame_vec = frame_tokens.mean(dim=1)  # [B, d_model]

        # 2. 计算帧内方差（检测模糊/噪声）
        token_variance = frame_tokens.var(dim=1)  # [B, d_model]
        variance_feature = token_variance.mean(dim=-1, keepdim=True)  # [B, 1]

        # 3. 计算帧间相似度（如果存在前一帧）
        if self.prev_frame_vec is None:
            # 第一帧，默认中等相似度
            similarity = torch.ones(batch_size, 1, device=frame_tokens.device) * 0.5
        else:
            # 计算余弦相似度
            prev_vec = self.prev_frame_vec.to(frame_vec.device)
            similarity = F.cosine_similarity(
                frame_vec, prev_vec, dim=-1, eps=self.eps
            ).unsqueeze(-1)  # [B, 1]

        # 4. 综合评分 (Cycle 1 fix: motion-neutral quality scoring)
        # Only penalize true artifacts: frame collapse (sim<0.05) or frozen frames (sim>0.99)
        # All normal motion levels (0.05 to 0.99 similarity) get quality=1.0
        # This prevents penalizing high-motion frames that caused dynamic_degree to drop

        # Collapse penalty: linear decrease from 1.0 at sim=0.05 to 0.0 at sim=-1.0
        collapse_penalty = torch.clamp((similarity - (-1.0)) / (0.05 - (-1.0)), 0.0, 1.0)
        # Frozen penalty: mild linear decrease from 1.0 at sim=0.99 to 0.3 at sim=1.0
        frozen_penalty = torch.clamp(1.0 - 0.7 * (similarity - 0.99) / (1.0 - 0.99), 0.3, 1.0)
        # Combined: use collapse_penalty below 0.05, frozen_penalty above 0.99, 1.0 in between
        sim_quality = torch.where(
            similarity < 0.05,
            collapse_penalty,
            torch.where(similarity > 0.99, frozen_penalty, torch.ones_like(similarity))
        )
        sim_quality = torch.clamp(sim_quality, 0.0, 1.0)

        # 方差应该在一个合理范围内（不太低=不模糊，不太高=不噪声）
        var_norm = torch.clamp(variance_feature / (variance_feature + 0.1), 0.0, 1.0)
        var_quality = 1.0 - torch.abs(var_norm - 0.5) * 2.0  # 峰值在 0.5
        var_quality = torch.clamp(var_quality, 0.0, 1.0)

        # 组合质量分数 - reduce sim_quality weight to allow more motion diversity
        quality = 0.3 * sim_quality + 0.7 * var_quality

        # 更新前一帧向量（使用 EMA）
        if self.prev_frame_vec is None:
            self.prev_frame_vec = frame_vec.detach().clone()
        else:
            self.prev_frame_vec = 0.7 * self.prev_frame_vec + 0.3 * frame_vec.detach()

        if squeeze_output:
            quality = quality.squeeze(0)

        return quality

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """前向接口，与 score_frame 相同"""
        return self.score_frame(frame_tokens)


class FloatTokenBank(nn.Module):
    """
    Float Token Bank - 动态更新机制

    使用 EMA（指数移动平均）压缩历史信息到固定数量的 slots 中。
    当 KV cache 滑窗丢弃旧帧时，这些帧的信息会被压缩到 float tokens。

    Args:
        num_slots: Float token 的数量
        d_model: 模型维度
        alpha: EMA 更新系数 (0-1)，越大更新越快
        update_interval: 更新间隔（每 N 帧更新一次）
    """

    def __init__(
        self,
        num_slots: int = 4,
        d_model: int = 2048,
        alpha: float = 0.2,
        update_interval: int = 1,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.alpha = alpha
        self.update_interval = update_interval
        self.eps = eps

        # Float token slots - 可学习的参数
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)

        # 非参数状态
        self.register_buffer('slot_cursor', torch.tensor(0, dtype=torch.long))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('slot_usage', torch.zeros(num_slots))  # 记录各 slot 使用频率

    def reset(self):
        """重置状态"""
        self.slot_cursor.zero_()
        self.update_count.zero_()
        self.slot_usage.zero_()
        nn.init.normal_(self.slots, std=0.02)

    def update(
        self,
        evicted_tokens: torch.Tensor,
        quality_score: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        使用被驱逐的 tokens 更新 float token bank

        Args:
            evicted_tokens: 被驱逐的 tokens [B, num_evicted, d_model] 或 [num_evicted, d_model]
            quality_score: 质量分数 [B, 1] 或 scalar，用于加权更新

        Returns:
            stats: 包含更新统计信息的字典
        """
        self.update_count += 1

        # 检查是否需要更新（根据 update_interval）
        if self.update_count % self.update_interval != 0:
            return {
                'updated': False,
                'cursor': self.slot_cursor.item(),
                'slot_norms': self.slots.norm(dim=-1).detach()
            }

        # 确保是 3D
        if evicted_tokens.dim() == 2:
            evicted_tokens = evicted_tokens.unsqueeze(0)

        batch_size = evicted_tokens.shape[0]

        # 1. 池化被驱逐的 tokens
        # 对所有被驱逐的 tokens 做平均池化，得到帧向量
        frame_vec = evicted_tokens.mean(dim=1)  # [B, d_model]

        # 2. 如果有多个 batch，再做一次平均
        if batch_size > 1:
            frame_vec = frame_vec.mean(dim=0, keepdim=True)  # [1, d_model]

        # 3. EMA 更新
        i = self.slot_cursor.item()

        # 计算有效 alpha（考虑质量分数）
        if quality_score is not None:
            if isinstance(quality_score, torch.Tensor):
                # 如果是 batch，取平均
                effective_alpha = self.alpha * quality_score.mean().item()
            else:
                effective_alpha = self.alpha * quality_score
        else:
            effective_alpha = self.alpha

        effective_alpha = max(0.05, min(0.95, effective_alpha))  # 限制范围

        # 执行 EMA 更新
        with torch.no_grad():
            self.slots[i] = (
                effective_alpha * frame_vec.squeeze(0) +
                (1 - effective_alpha) * self.slots[i]
            )
            self.slot_usage[i] += 1

        # 4. 轮转指针
        self.slot_cursor = (self.slot_cursor + 1) % self.num_slots

        return {
            'updated': True,
            'cursor': i,
            'next_cursor': self.slot_cursor.item(),
            'alpha': effective_alpha,
            'slot_norms': self.slots.norm(dim=-1).detach(),
            'frame_vec_norm': frame_vec.norm().item()
        }

    def get_tokens(self) -> torch.Tensor:
        """
        获取当前所有 float tokens

        Returns:
            tokens: [num_slots, d_model]
        """
        return self.slots

    def get_stats(self) -> Dict[str, torch.Tensor]:
        """获取统计信息用于调试"""
        return {
            'slot_norms': self.slots.norm(dim=-1).detach(),
            'slot_usage': self.slot_usage.clone(),
            'cursor': self.slot_cursor.item(),
            'update_count': self.update_count.item(),
            'mean_slot_norm': self.slots.norm(dim=-1).mean().item()
        }


class HierarchicalFloatBank(nn.Module):
    """
    分层 Float Token Bank

    同时维护三组不同时间尺度的 float tokens：
    - 短期（Short）: 快速响应，每帧更新
    - 中期（Mid）: 稳定场景，每 30 帧更新
    - 长期（Long）: 锁定布局，每 90 帧更新

    Args:
        d_model: 模型维度
        num_slots_short/mid/long: 各层 slot 数量
        alpha_short/mid/long: 各层 EMA 系数
        update_interval_short/mid/long: 各层更新间隔
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_slots_short: int = 4,
        num_slots_mid: int = 4,
        num_slots_long: int = 4,
        alpha_short: float = 0.3,
        alpha_mid: float = 0.15,
        alpha_long: float = 0.05,
        update_interval_short: int = 1,
        update_interval_mid: int = 30,
        update_interval_long: int = 90,
        use_quality_scorer: bool = True,
        eps: float = 1e-6
    ):
        super().__init__()
        self.d_model = d_model
        self.use_quality_scorer = use_quality_scorer

        # 质量评分器
        if use_quality_scorer:
            self.quality_scorer = FrameQualityScorer(d_model, eps)
        else:
            self.quality_scorer = None

        # 三层 float bank
        self.bank_short = FloatTokenBank(
            num_slots=num_slots_short,
            d_model=d_model,
            alpha=alpha_short,
            update_interval=update_interval_short,
            eps=eps
        )

        self.bank_mid = FloatTokenBank(
            num_slots=num_slots_mid,
            d_model=d_model,
            alpha=alpha_mid,
            update_interval=update_interval_mid,
            eps=eps
        )

        self.bank_long = FloatTokenBank(
            num_slots=num_slots_long,
            d_model=d_model,
            alpha=alpha_long,
            update_interval=update_interval_long,
            eps=eps
        )

        self.total_slots = num_slots_short + num_slots_mid + num_slots_long

    def reset(self):
        """重置所有层的状态"""
        self.bank_short.reset()
        self.bank_mid.reset()
        self.bank_long.reset()
        if self.quality_scorer is not None:
            self.quality_scorer.reset()

    def update(
        self,
        evicted_tokens: torch.Tensor,
        frame_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict]:
        """
        更新所有层的 float tokens

        Args:
            evicted_tokens: 被驱逐的 tokens
            frame_tokens: 当前帧的完整 tokens（用于质量评分，可选）

        Returns:
            stats: 各层更新统计信息
        """
        # 计算质量分数
        quality_score = None
        if self.quality_scorer is not None and frame_tokens is not None:
            quality_score = self.quality_scorer(frame_tokens)

        # 更新各层
        stats_short = self.bank_short.update(evicted_tokens, quality_score)
        stats_mid = self.bank_mid.update(evicted_tokens, quality_score)
        stats_long = self.bank_long.update(evicted_tokens, quality_score)

        return {
            'short': stats_short,
            'mid': stats_mid,
            'long': stats_long,
            'quality_score': quality_score.mean().item() if isinstance(quality_score, torch.Tensor) else quality_score
        }

    def get_all_tokens(self) -> torch.Tensor:
        """
        获取所有层的 float tokens

        Returns:
            tokens: [total_slots, d_model]
        """
        tokens_short = self.bank_short.get_tokens()
        tokens_mid = self.bank_mid.get_tokens()
        tokens_long = self.bank_long.get_tokens()

        return torch.cat([tokens_short, tokens_mid, tokens_long], dim=0)

    def get_stats(self) -> Dict[str, Dict]:
        """获取所有层的统计信息"""
        return {
            'short': self.bank_short.get_stats(),
            'mid': self.bank_mid.get_stats(),
            'long': self.bank_long.get_stats()
        }


def apply_rope_with_float_tokens(
    x: torch.Tensor,
    float_tokens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
    rope_apply_fn = None
) -> torch.Tensor:
    """
    对普通帧和 float tokens 分别应用 RoPE

    关键改进：float tokens 使用锚定位置（start_frame=0），不随时间漂移

    Args:
        x: 普通帧 tokens [B, seq_len, num_heads, head_dim]
        float_tokens: float tokens [num_float_tokens, num_heads, head_dim]
        grid_sizes: 网格尺寸 [B, 3]
        freqs: RoPE 频率
        start_frame: 当前 block 的起始帧
        rope_apply_fn: 外部的 rope_apply 函数

    Returns:
        roped_x: 应用 RoPE 后的结果 [B, seq_len + num_float_tokens, num_heads, head_dim]
    """
    if rope_apply_fn is None:
        from wan.modules.model import rope_apply
        rope_apply_fn = rope_apply

    # 分离普通帧和 float tokens
    x_frames = x

    # 扩展 float tokens 到 batch 维度
    batch_size = x.shape[0]
    float_tokens_expanded = float_tokens.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # 普通帧：正常 RoPE，使用 start_frame
    roped_frames = rope_apply_fn(x_frames, grid_sizes, freqs, start_frame).type_as(x)

    # Float tokens：锚定在当前 block 起点（start_frame=0）
    # 为 float tokens 创建虚拟的 grid_sizes（1帧）
    float_grid_sizes = torch.ones_like(grid_sizes)
    float_grid_sizes[:, 0] = 1  # 1 frame
    float_grid_sizes[:, 1:] = 1  # 1x1 spatial

    roped_float = rope_apply_fn(float_tokens_expanded, float_grid_sizes, freqs, start_frame=0).type_as(x)

    # 拼接
    return torch.cat([roped_frames, roped_float], dim=1)


def causal_rope_apply_with_float_tokens(
    x: torch.Tensor,
    float_tokens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0
) -> torch.Tensor:
    """
    Causal RoPE 应用，支持 float tokens

    这是 apply_rope_with_float_tokens 的 causal 版本
    """
    from wan.modules.causal_model import causal_rope_apply

    # 分离普通帧和 float tokens
    x_frames = x

    # 扩展 float tokens 到 batch 维度
    batch_size = x.shape[0]
    float_tokens_expanded = float_tokens.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # 普通帧：正常 causal RoPE
    roped_frames = causal_rope_apply(x_frames, grid_sizes, freqs, start_frame).type_as(x)

    # Float tokens：锚定位置
    float_grid_sizes = torch.ones_like(grid_sizes)
    float_grid_sizes[:, 0] = 1
    float_grid_sizes[:, 1:] = 1

    roped_float = causal_rope_apply(float_tokens_expanded, float_grid_sizes, freqs, start_frame=0).type_as(x)

    return torch.cat([roped_frames, roped_float], dim=1)


class FloatTokenDebugger:
    """
    Float Token 调试工具

    用于监控和分析 float tokens 的状态
    """

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.step_count = 0
        self.history = {
            'quality_scores': [],
            'slot_norms_short': [],
            'slot_norms_mid': [],
            'slot_norms_long': [],
            'update_counts': []
        }

    def log(self, stats: Dict):
        """记录统计信息"""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            print(f"\n=== Float Token Stats (Step {self.step_count}) ===")

            if 'quality_score' in stats:
                print(f"  Quality Score: {stats['quality_score']:.3f}")
                self.history['quality_scores'].append(stats['quality_score'])

            for level in ['short', 'mid', 'long']:
                if level in stats and isinstance(stats[level], dict):
                    if 'slot_norms' in stats[level]:
                        norms = stats[level]['slot_norms']
                        print(f"  {level.capitalize()} Slot Norms: {norms.tolist()}")
                        self.history[f'slot_norms_{level}'].append(norms.cpu().numpy())

    def plot_history(self, save_path: Optional[str] = None):
        """绘制历史记录"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 质量分数
            if self.history['quality_scores']:
                axes[0, 0].plot(self.history['quality_scores'])
                axes[0, 0].set_title('Quality Scores')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_ylim([0, 1])

            # 各层 slot norms
            for i, level in enumerate(['short', 'mid', 'long']):
                ax = axes[0, 1] if i == 0 else (axes[1, 0] if i == 1 else axes[1, 1])
                key = f'slot_norms_{level}'
                if self.history[key]:
                    data = self.history[key]
                    ax.plot(data)
                    ax.set_title(f'{level.capitalize()} Slot Norms')
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Norm')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.savefig('float_token_stats.png')
                print("Plot saved to float_token_stats.png")

            plt.close()
        except ImportError:
            print("matplotlib not available, skipping plot")

    def print_summary(self):
        """打印摘要统计"""
        print("\n=== Float Token Summary ===")

        if self.history['quality_scores']:
            scores = self.history['quality_scores']
            print(f"Quality Score: mean={sum(scores)/len(scores):.3f}, "
                  f"min={min(scores):.3f}, max={max(scores):.3f}")

        for level in ['short', 'mid', 'long']:
            key = f'slot_norms_{level}'
            if self.history[key]:
                norms = self.history[key]
                mean_norm = sum(n.mean() for n in norms) / len(norms)
                print(f"{level.capitalize()} Mean Slot Norm: {mean_norm:.3f}")


class FloatKVSlot(nn.Module):
    """
    Float KV Bank V2 - 直接存储K/V对，使用slot-attention更新
    
    解决了现有FloatTokenBank的双重投影问题：
    - 旧版：存储压缩的input-space向量，注意力时再次投影成K/V
    - 新版：直接存储K/V对，绕过input-space中间表示
    
    Args:
        num_slots: slot数量
        num_heads: 注意力头数
        head_dim: 每个头的维度
        alpha: EMA更新系数
        update_interval: 更新间隔
        use_dynamic_intervals: 是否使用动态更新间隔
        dynamic_interval_min_factor: 动态间隔最小因子
        dynamic_interval_max_factor: 动态间隔最大因子
    """
    
    def __init__(
        self,
        num_slots: int = 4,
        num_heads: int = 16,
        head_dim: int = 128,
        alpha: float = 0.2,
        update_interval: int = 1,
        eps: float = 1e-6,
        use_fifo_slots: bool = True,  # FIFO slot assignment for temporal diversity
        use_dynamic_intervals: bool = False,
        dynamic_interval_min_factor: float = 0.5,
        dynamic_interval_max_factor: float = 3.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.alpha = alpha
        self.base_update_interval = update_interval
        self.eps = eps
        self.use_fifo_slots = use_fifo_slots
        self.use_dynamic_intervals = use_dynamic_intervals

        # 直接存储K/V对 - shape: [num_slots, num_heads, head_dim]
        self.register_buffer('slots_k', torch.zeros(num_slots, num_heads, head_dim))
        self.register_buffer('slots_v', torch.zeros(num_slots, num_heads, head_dim))

        # 状态标记
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
        # FIFO slot pointer: which slot to write next (for temporal diversity)
        self.register_buffer('slot_write_ptr', torch.tensor(0, dtype=torch.long))
        # Track which slots have been written at least once (for first-write direct assignment)
        self.register_buffer('slot_written', torch.zeros(num_slots, dtype=torch.bool))

        # Content-Adaptive Alpha: 存储前一次的evicted_k平均值
        self.register_buffer('prev_evicted_k_avg', torch.zeros(num_heads, head_dim))
        self.register_buffer('has_prev', torch.tensor(False, dtype=torch.bool))

        # Cycle 1: Quality-Adaptive Slot Confidence Tracking
        # slot_confidence[i] tracks EMA of quality scores for slot i [0, 1]
        # Higher confidence = slot has received consistently high-quality updates = more stable
        self.register_buffer('slot_confidence', torch.zeros(num_slots))
        # slot_staleness[i] counts eviction steps since slot i was last updated
        # When staleness > staleness_threshold, force update even if interval not reached
        self.register_buffer('slot_staleness', torch.zeros(num_slots, dtype=torch.long))
        self.staleness_threshold = 50  # force update after 50 eviction steps without update

        # Cycle 7: Momentum-Based EMA Update
        # slots_k_prev tracks the previous slot values for momentum computation
        # momentum * (slot_old - slot_prev) is added to the EMA update
        self.register_buffer('slots_k_prev', torch.zeros(num_slots, num_heads, head_dim))
        self.register_buffer('slots_v_prev', torch.zeros(num_slots, num_heads, head_dim))
        self.momentum_factor = 0.1  # momentum coefficient (small to avoid instability)

        # Cycle 8: Slot Norm Clipping
        # Cycle 6 update: Reduce max_slot_norm from 3x to 2x for tighter V-value control
        # Prevents slot norms from exploding during long video generation
        self.max_slot_norm = head_dim ** 0.5 * 2.0  # Cycle 6: 2x expected norm (was 3x)

        # Cycle 10: KV Norm Quality Proxy
        # Track running stats of healthy KV norms to detect anomalous frames
        self.register_buffer('kv_norm_ema', torch.tensor(0.0))
        self.register_buffer('kv_norm_initialized', torch.tensor(False, dtype=torch.bool))
        self.kv_norm_alpha = 0.1  # EMA alpha for norm tracking
        
        # Dynamic interval scheduler (only created if needed)
        if use_dynamic_intervals:
            self.interval_scheduler = DynamicIntervalScheduler(
                base_interval=update_interval,
                min_factor=dynamic_interval_min_factor,
                max_factor=dynamic_interval_max_factor
            )
        else:
            self.interval_scheduler = None
    
    def reset(self):
        """重置states状态"""
        self.slots_k.zero_()
        self.slots_v.zero_()
        self.initialized.fill_(False)
        self.update_count.zero_()
        self.prev_evicted_k_avg.zero_()
        self.has_prev.fill_(False)
        self.slot_write_ptr.zero_()
        self.slot_written.fill_(False)
        # Cycle 1: reset confidence and staleness tracking
        self.slot_confidence.zero_()
        self.slot_staleness.zero_()
        # Cycle 7: reset momentum buffers
        self.slots_k_prev.zero_()
        self.slots_v_prev.zero_()
        # Cycle 10: reset KV norm tracking
        self.kv_norm_ema.zero_()
        self.kv_norm_initialized.fill_(False)
        if self.interval_scheduler is not None:
            self.interval_scheduler.reset()

    def get_slot_effective_alpha(self, slot_idx: int, quality_score: float) -> float:
        """
        Cycle 1: Compute confidence-adjusted effective alpha for a specific slot.

        High-confidence slots (consistently good updates) are more stable -> lower alpha.
        Quality score scales how much the new info is trusted.

        Args:
            slot_idx: Index of the slot being updated
            quality_score: Quality score of the current evicted frame [0, 1]

        Returns:
            effective_alpha: Adjusted alpha value in [0.05, 0.95]
        """
        confidence = self.slot_confidence[slot_idx].item()
        # stability_factor: 1.0 when no confidence, 0.5 when fully confident
        # High-confidence slots receive smaller updates (more stable)
        stability_factor = 1.0 - 0.5 * confidence
        effective_alpha = self.alpha * quality_score * stability_factor
        return max(0.05, min(0.95, effective_alpha))
    
    def _compute_adaptive_alpha(self, evicted_k: torch.Tensor) -> float:
        """
        计算内容自适应的alpha值
        
        关键逻辑：当连续边缘化差异大时(低余弦相似度=场景变化)，使用高alpha快速捕捉新内容
        当连续边缘化相似时(高余弦相似度=稳定场景)，使用低alpha平滑累积
        
        Args:
            evicted_k: 被驱逐的key [N, num_heads, head_dim]
            
        Returns:
            alpha_adaptive: 自适应alpha值
        """
        with torch.no_grad():
            # 对tokens取平均 -> [num_heads, head_dim]
            ek = evicted_k.mean(dim=0)  # [H, D]
            
            # 如果没有前一次的记录，存储当前值并返回基础alpha
            if not self.has_prev:
                self.prev_evicted_k_avg.copy_(ek)
                self.has_prev.fill_(True)
                return self.alpha
            
            # 计算余弦相似度: 对每个head计算后取平均
            # cosine_similarity: [H, D] vs [H, D] -> [H]
            cosine_sim = F.cosine_similarity(
                ek, self.prev_evicted_k_avg, dim=-1, eps=self.eps
            ).mean()  # 对heads取平均得到标量
            
            # 计算自适应alpha: 场景变化时增大，稳定时减小
            # alpha_adaptive = alpha_base * (1 + 2 * (1 - cosine_sim))
            alpha_adaptive = self.alpha * (1.0 + 2.0 * (1.0 - cosine_sim.item()))

            # Cycle 3: Cap alpha_max at 0.5 to prevent volatile slot overwrite during scene changes
            # Prior: min(alpha_base * 3, 0.9) — at alpha_base=0.5, allowed 90% slot overwrite
            # Now: min(alpha_base * 2, 0.5) — at most 50% blending even during fast scene changes
            alpha_min = self.alpha * 0.5
            alpha_max = min(self.alpha * 2.0, 0.5)
            alpha_adaptive = max(alpha_min, min(alpha_max, alpha_adaptive))
            
            # 更新prev_evicted_k_avg为滚动平均(alpha=0.5)
            self.prev_evicted_k_avg.mul_(0.5).add_(ek, alpha=0.5)
            
            return alpha_adaptive
    
    def update(
        self,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        quality_score: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        使用被驱逐的K/V更新float KV bank
        
        Args:
            evicted_k: 被驱逐的key [B, N, num_heads, head_dim] 或 [N, num_heads, head_dim]
            evicted_v: 被驱逐的value [B, N, num_heads, head_dim] 或 [N, num_heads, head_dim]
            quality_score: 质量分数，用于加权更新
        
        Returns:
            stats: 更新统计信息
        """
        # Guard: if num_slots=0, this bank is disabled
        if self.num_slots == 0:
            return {'updated': False, 'disabled': True}

        self.update_count += 1

        # 确定当前更新间隔
        if self.use_dynamic_intervals and self.interval_scheduler is not None:
            # 计算内容相似度用于动态间隔
            if self.has_prev:
                with torch.no_grad():
                    ek = evicted_k.mean(dim=0) if evicted_k.dim() == 3 else evicted_k.mean(dim=0)
                    cosine_sim = F.cosine_similarity(
                        ek, self.prev_evicted_k_avg, dim=-1, eps=self.eps
                    ).mean().item()
            else:
                cosine_sim = 0.0  # 默认中等相似度
            current_interval = self.interval_scheduler.get_interval(cosine_sim)
        else:
            current_interval = self.base_update_interval

        # Cycle 1: Increment staleness for all slots each eviction step
        if self.initialized:
            self.slot_staleness += 1

        # Cycle 1: Check if any slot is too stale and needs a forced update
        force_update_due_to_staleness = (
            self.initialized and
            (self.slot_staleness >= self.staleness_threshold).any()
        )

        # 检查更新间隔 (skip interval check if staleness forces update)
        if not force_update_due_to_staleness and self.update_count % current_interval != 0:
            return {
                'updated': False,
                'update_count': self.update_count.item(),
                'current_interval': current_interval,
                'slot_k_norms': self.slots_k.norm(dim=-1).mean(dim=-1).detach()
            }
        
        # 确保是4D [B, N, H, D]
        if evicted_k.dim() == 3:
            evicted_k = evicted_k.unsqueeze(0)
            evicted_v = evicted_v.unsqueeze(0)
        
        batch_size = evicted_k.shape[0]
        
        # 对batch取平均 -> [N, num_heads, head_dim]
        ek = evicted_k.mean(dim=0)  # [N, H, D]
        ev = evicted_v.mean(dim=0)  # [N, H, D]
        
        with torch.no_grad():
            # Eviction Quality Gate: reject frames with anomalous KV norms
            # (too low = collapsed/blank frames, too high = unstable/noise frames)
            ek_norm = ek.norm(dim=-1).mean().item()  # Average norm per token

            # Cycle 10: Update KV norm EMA for quality tracking
            if not self.kv_norm_initialized:
                self.kv_norm_ema.fill_(ek_norm)
                self.kv_norm_initialized.fill_(True)
            else:
                self.kv_norm_ema.mul_(1 - self.kv_norm_alpha).add_(ek_norm * self.kv_norm_alpha)

            # Cycle 10: Use KV norm deviation as quality signal
            # Frames deviating >3x from EMA are likely anomalous
            norm_ema = self.kv_norm_ema.item()
            if norm_ema > 0:
                norm_deviation = abs(ek_norm - norm_ema) / (norm_ema + 1e-6)
                # Scale quality by deviation: max quality at 0 deviation, 0 quality at 3x deviation
                kv_quality = max(0.1, 1.0 - norm_deviation / 3.0)
                # Blend with provided quality score
                quality_score = quality_score * kv_quality

            if self.initialized:
                # Reference scale: use current slot norms as expected range
                slot_norm = self.slots_k.norm(dim=-1).mean().item()
                if slot_norm > 0:
                    norm_ratio = ek_norm / (slot_norm + 1e-6)
                    # Reject if norm is too extreme (< 0.1x or > 10x reference)
                    if norm_ratio < 0.1 or norm_ratio > 10.0:
                        return {
                            'updated': False,
                            'rejected': True,
                            'norm_ratio': norm_ratio,
                            'slot_k_norms': self.slots_k.norm(dim=-1).mean(dim=-1).detach()
                        }

            # 首次初始化：将evicted tokens分块成num_slots组
            if not self.initialized:
                N = ek.shape[0]
                if N >= self.num_slots:
                    # 将N个tokens均匀分配到num_slots个slot
                    tokens_per_slot = N // self.num_slots
                    for i in range(self.num_slots):
                        start_idx = i * tokens_per_slot
                        if i == self.num_slots - 1:
                            end_idx = N  # 最后一个slot取所有剩余
                        else:
                            end_idx = (i + 1) * tokens_per_slot
                        self.slots_k[i] = ek[start_idx:end_idx].mean(dim=0)
                        self.slots_v[i] = ev[start_idx:end_idx].mean(dim=0)
                else:
                    # 如果N < num_slots，轮询填充
                    for i in range(self.num_slots):
                        idx = i % N
                        self.slots_k[i] = ek[idx]
                        self.slots_v[i] = ev[idx]
                
                self.initialized.fill_(True)
                # Bug fix (Cycle 1 follow-up): Mark all initialized slots as written
                # Previously, slot_written stayed False after init, causing all subsequent
                # FIFO updates to use direct assignment (overwriting EMA entirely)
                self.slot_written.fill_(True)
                # Initialize prev buffers from initial slot values
                self.slots_k_prev.copy_(self.slots_k)
                self.slots_v_prev.copy_(self.slots_v)
                return {
                    'updated': True,
                    'initialized': True,
                    'slot_k_norms': self.slots_k.norm(dim=-1).mean(dim=-1).detach()
                }
            
            # Compute the new KV representation from evicted tokens
            # Average eviction over tokens for each head
            new_k_raw = ek.mean(dim=0, keepdim=True)  # [1, H, D]
            new_v_raw = ev.mean(dim=0, keepdim=True)  # [1, H, D]

            # 计算内容自适应alpha
            adaptive_alpha = self._compute_adaptive_alpha(ek)

            # 结合质量分数计算有效alpha (base combined alpha)
            base_eff_alpha = max(0.05, min(0.95, adaptive_alpha * quality_score))

            if self.use_fifo_slots:
                # FIFO slot assignment: write to next slot in ring buffer
                # This ensures each slot holds a DIFFERENT temporal snapshot
                # (temporal diversity), rather than all slots chasing the same eviction

                # Cycle 1: If staleness forced the update, target the stalest slot instead
                if force_update_due_to_staleness and (self.slot_staleness >= self.staleness_threshold).any():
                    slot_idx = self.slot_staleness.argmax().item()
                elif self.initialized and self.num_slots > 1:
                    # Cycle 2: Diversity-Aware Slot Selection
                    # Find the slot with content MOST DIFFERENT from new_k_raw
                    # (maximize temporal diversity across slots)
                    new_k_mean = new_k_raw[0].mean(dim=0)  # [head_dim]
                    slots_k_mean = self.slots_k.mean(dim=1)  # [num_slots, head_dim]
                    # Cosine similarity between new content and each slot
                    cos_sims = F.cosine_similarity(
                        new_k_mean.unsqueeze(0),  # [1, head_dim]
                        slots_k_mean,             # [num_slots, head_dim]
                        dim=-1
                    )  # [num_slots]
                    # Target the most dissimilar slot (maximizes diversity)
                    # But only among written slots; fall back to FIFO for unwritten
                    if self.slot_written.any():
                        written_indices = self.slot_written.nonzero(as_tuple=True)[0]
                        # Among written slots, pick most dissimilar
                        written_sims = cos_sims[written_indices]
                        most_dissimilar_local = written_sims.argmin().item()
                        slot_idx = written_indices[most_dissimilar_local].item()
                    else:
                        slot_idx = self.slot_write_ptr.item() % self.num_slots
                else:
                    slot_idx = self.slot_write_ptr.item() % self.num_slots

                # Cycle 1: Use confidence-adjusted alpha for this specific slot
                eff_alpha = self.get_slot_effective_alpha(slot_idx, quality_score)

                # Cycle 10: Skip redundant writes — if new content is too similar to
                # the target slot, skip the update to preserve temporal diversity
                # This prevents slots from all converging to the same recent scene
                if self.slot_written[slot_idx]:
                    target_slot_k = self.slots_k[slot_idx]  # [H, D]
                    new_k_mean = new_k_raw[0].mean(dim=0)   # [D] (mean over heads)
                    target_k_mean = target_slot_k.mean(dim=0)
                    redundancy_sim = F.cosine_similarity(
                        new_k_mean.unsqueeze(0), target_k_mean.unsqueeze(0), dim=-1
                    ).item()
                    # Skip if >95% similar (near-duplicate frame)
                    if redundancy_sim > 0.95 and self.slot_written.sum() == self.num_slots:
                        return {
                            'updated': False,
                            'skipped_redundant': True,
                            'redundancy_sim': redundancy_sim,
                            'slot_k_norms': self.slots_k.norm(dim=-1).mean(dim=-1).detach()
                        }

                if not self.slot_written[slot_idx]:
                    # First write to this slot: direct assignment (no blending with zero init)
                    self.slots_k[slot_idx].copy_(new_k_raw[0])
                    self.slots_v[slot_idx].copy_(new_v_raw[0])
                    self.slot_written[slot_idx] = True
                    # Initialize prev buffers
                    self.slots_k_prev[slot_idx].copy_(new_k_raw[0])
                    self.slots_v_prev[slot_idx].copy_(new_v_raw[0])
                else:
                    # Cycle 7: Momentum-Based EMA Update
                    # Save previous slot values for momentum computation
                    k_old = self.slots_k[slot_idx].clone()
                    v_old = self.slots_v[slot_idx].clone()

                    # EMA blend
                    new_k = k_old * (1 - eff_alpha) + new_k_raw[0] * eff_alpha
                    new_v = v_old * (1 - eff_alpha) + new_v_raw[0] * eff_alpha

                    # Add momentum: push in direction of recent change
                    momentum_k = k_old - self.slots_k_prev[slot_idx]
                    momentum_v = v_old - self.slots_v_prev[slot_idx]
                    new_k = new_k + self.momentum_factor * momentum_k
                    new_v = new_v + self.momentum_factor * momentum_v

                    # Cycle 8: Slot Norm Clipping
                    # Prevent norms from exploding during long video
                    k_norms = new_k.norm(dim=-1, keepdim=True)  # [H, 1]
                    v_norms = new_v.norm(dim=-1, keepdim=True)
                    new_k = torch.where(
                        k_norms > self.max_slot_norm,
                        new_k / k_norms * self.max_slot_norm,
                        new_k
                    )
                    new_v = torch.where(
                        v_norms > self.max_slot_norm,
                        new_v / v_norms * self.max_slot_norm,
                        new_v
                    )

                    # Update prev before overwriting current
                    self.slots_k_prev[slot_idx].copy_(k_old)
                    self.slots_v_prev[slot_idx].copy_(v_old)

                    self.slots_k[slot_idx].copy_(new_k)
                    self.slots_v[slot_idx].copy_(new_v)

                # Cycle 1: Update confidence and reset staleness for this slot
                self.slot_confidence[slot_idx] = (
                    0.9 * self.slot_confidence[slot_idx] + 0.1 * quality_score
                )
                self.slot_staleness[slot_idx] = 0

                if not force_update_due_to_staleness:
                    self.slot_write_ptr.add_(1)
            else:
                eff_alpha = base_eff_alpha
                # Slot Attention: each slot selects relevant evicted tokens via softmax
                slots_k_avg = self.slots_k.mean(dim=1)  # [K, D]
                ek_avg = ek.mean(dim=1)  # [N, D]

                # 计算attention logits: [K, N]
                attn_logits = torch.einsum('kd,nd->kn', slots_k_avg, ek_avg) * (self.head_dim ** -0.5)
                attn_weights = F.softmax(attn_logits, dim=-1)  # [K, N]

                # 计算新的K/V: 用attention weights加权求和
                new_k = torch.einsum('kn,nhd->khd', attn_weights, ek)
                new_v = torch.einsum('kn,nhd->khd', attn_weights, ev)

                # EMA更新 (all slots updated with confidence-adjusted alpha for each slot)
                for s_idx in range(self.num_slots):
                    slot_alpha = self.get_slot_effective_alpha(s_idx, quality_score)
                    self.slots_k[s_idx].mul_(1 - slot_alpha).add_(new_k[s_idx], alpha=slot_alpha)
                    self.slots_v[s_idx].mul_(1 - slot_alpha).add_(new_v[s_idx], alpha=slot_alpha)
                    # Update confidence and staleness for each slot
                    self.slot_confidence[s_idx] = (
                        0.9 * self.slot_confidence[s_idx] + 0.1 * quality_score
                    )
                    self.slot_staleness[s_idx] = 0
        
        return {
            'updated': True,
            'initialized': True,
            'alpha': eff_alpha,
            'adaptive_alpha': adaptive_alpha,
            'quality_score': quality_score,
            'force_update': force_update_due_to_staleness,
            'slot_confidences': self.slot_confidence.detach().clone(),
            'slot_k_norms': self.slots_k.norm(dim=-1).mean(dim=-1).detach()
        }

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取已写入slot的K/V对 (只返回已初始化的slot，过滤零值)

        Cycle 3: Apply slot normalization to scale K/V norms to match the
        expected KV cache norm scale. This prevents float token slots with
        large norms from dominating the softmax during attention.

        Returns:
            slots_k: [num_written_slots, num_heads, head_dim] (norm-scaled)
            slots_v: [num_written_slots, num_heads, head_dim]
        """
        if self.use_fifo_slots and hasattr(self, 'slot_written'):
            # Only return slots that have been written to at least once
            written_mask = self.slot_written  # [num_slots] bool
            if written_mask.any():
                sk = self.slots_k[written_mask]
                sv = self.slots_v[written_mask]
                # Cycle 3: Normalize K slots to unit norm per head, preserve direction
                # This equalizes float token influence regardless of slot magnitude
                k_norms = sk.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [S, H, 1]
                target_norm = (self.head_dim ** 0.5)  # match expected scale
                sk_normalized = sk / k_norms * target_norm

                # Cycle 4: Apply temporal decay weighting based on staleness
                # Stale slots (old information) are downweighted
                # decay_factor[i] = exp(-staleness[i] / decay_tau)
                # where decay_tau = staleness_threshold / 2 = 25
                staleness_written = self.slot_staleness[written_mask].float()
                # Cycle 4: Increase decay_tau from staleness/2=25 to staleness*2=100
                # Prior: tau=25 → after 25 evictions (~4s), V weight ≈ 0.37 (too aggressive)
                # Now: tau=100 → after 100 evictions (~16s), V weight ≈ 0.37 (gentler decay)
                decay_tau = max(1.0, self.staleness_threshold * 2.0)
                decay_weights = torch.exp(-staleness_written / decay_tau)  # [S]
                # Apply decay to V values (K is already norm-scaled, don't double-scale)
                sv_decayed = sv * decay_weights.unsqueeze(-1).unsqueeze(-1)  # [S, H, D]

                return sk_normalized, sv_decayed
            else:
                # No slots written yet - return empty (caller should check is_ready())
                return self.slots_k[:0], self.slots_v[:0]
        # Non-FIFO mode: guard against returning zero-initialized slots
        if not self.initialized:
            return self.slots_k[:0], self.slots_v[:0]
        # Cycle 3+4: Also normalize and decay in non-FIFO mode
        k_norms = self.slots_k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        target_norm = (self.head_dim ** 0.5)
        sk_normalized = self.slots_k / k_norms * target_norm
        # Cycle 4: Apply temporal decay in non-FIFO mode
        # Cycle 4 update: use staleness*2 for decay_tau (gentler decay)
        decay_tau = max(1.0, self.staleness_threshold * 2.0)
        decay_weights = torch.exp(-self.slot_staleness.float() / decay_tau)
        sv_decayed = self.slots_v * decay_weights.unsqueeze(-1).unsqueeze(-1)
        return sk_normalized, sv_decayed


class HierarchicalFloatKVBank(nn.Module):
    """
    分层Float KV Bank V2 - 使用FloatKVSlot
    
    与HierarchicalFloatBank概念相同，但直接存储KV对而非input-space向量。
    
    Args:
        num_heads: 注意力头数
        head_dim: 每个头的维度
        num_slots_short/mid/long: 各层slot数量
        alpha_short/mid/long: 各层EMA系数
        update_interval_short/mid/long: 各层更新间隔
        use_quality_scorer: 是否使用质量评分
        use_temporal_coherence: 是否使用时间一致性评分
        use_progressive_activation: 是否使用渐进式银行激活
        use_dynamic_intervals: 是否使用动态更新间隔
        progressive_warmup_frames: 渐进式激活的warmup帧数
    """
    
    def __init__(
        self,
        num_heads: int = 16,
        head_dim: int = 128,
        num_slots_short: int = 4,
        num_slots_mid: int = 4,
        num_slots_long: int = 4,
        alpha_short: float = 0.3,
        alpha_mid: float = 0.15,
        alpha_long: float = 0.05,
        update_interval_short: int = 1,
        update_interval_mid: int = 10,
        update_interval_long: int = 30,
        use_quality_scorer: bool = True,
        use_temporal_coherence: bool = False,
        use_progressive_activation: bool = False,
        use_dynamic_intervals: bool = False,
        progressive_warmup_frames: int = 300,
        coherence_history_size: int = 30,
        dynamic_interval_min_factor: float = 0.5,
        dynamic_interval_max_factor: float = 3.0,
        # Cycle 5: Ultra-Long Tier for 1-minute video generation
        num_slots_ultra: int = 2,       # 2 slots for global theme preservation
        alpha_ultra: float = 0.02,      # very slow update (global average)
        update_interval_ultra: int = 100,  # update every 100 evictions (~1 min)
        use_ultra_long_tier: bool = True,  # Enable ultra-long tier by default
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_quality_scorer = use_quality_scorer
        self.use_temporal_coherence = use_temporal_coherence
        self.use_progressive_activation = use_progressive_activation

        # 创建三层FloatKVSlot
        self.bank_short = FloatKVSlot(
            num_slots=num_slots_short,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_short,
            update_interval=update_interval_short,
            eps=eps,
            use_dynamic_intervals=use_dynamic_intervals,
            dynamic_interval_min_factor=dynamic_interval_min_factor,
            dynamic_interval_max_factor=dynamic_interval_max_factor
        )

        self.bank_mid = FloatKVSlot(
            num_slots=num_slots_mid,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_mid,
            update_interval=update_interval_mid,
            eps=eps,
            use_dynamic_intervals=use_dynamic_intervals,
            dynamic_interval_min_factor=dynamic_interval_min_factor,
            dynamic_interval_max_factor=dynamic_interval_max_factor
        )

        self.bank_long = FloatKVSlot(
            num_slots=num_slots_long,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_long,
            update_interval=update_interval_long,
            eps=eps,
            use_dynamic_intervals=use_dynamic_intervals,
            dynamic_interval_min_factor=dynamic_interval_min_factor,
            dynamic_interval_max_factor=dynamic_interval_max_factor
        )

        # Cycle 5: Ultra-Long Tier for 1-minute video generation
        # Captures global scene theme across the full video
        self.use_ultra_long_tier = use_ultra_long_tier and num_slots_ultra > 0
        if self.use_ultra_long_tier:
            self.bank_ultra = FloatKVSlot(
                num_slots=num_slots_ultra,
                num_heads=num_heads,
                head_dim=head_dim,
                alpha=alpha_ultra,
                update_interval=update_interval_ultra,
                eps=eps,
                use_fifo_slots=False,  # Use slot-attention for global theme (no FIFO)
                use_dynamic_intervals=False  # Fixed interval for global theme
            )
        else:
            self.bank_ultra = None

        # 质量评分器（可选择类型）
        if use_temporal_coherence:
            self.coherence_scorer = TemporalCoherenceScorer(num_heads * head_dim, coherence_history_size)
            self.quality_scorer = None
        elif use_quality_scorer:
            self.quality_scorer = FrameQualityScorer(num_heads * head_dim, eps)
            self.coherence_scorer = None
        else:
            self.quality_scorer = None
            self.coherence_scorer = None

        # 渐进式激活
        if use_progressive_activation:
            self.progressive_activation = ProgressiveBankActivation(progressive_warmup_frames)
        else:
            self.progressive_activation = None

        ultra_slots = num_slots_ultra if self.use_ultra_long_tier else 0
        self.total_slots = num_slots_short + num_slots_mid + num_slots_long + ultra_slots
        self._alphas = {'short': alpha_short, 'mid': alpha_mid, 'long': alpha_long, 'ultra': alpha_ultra}
    
    def reset(self):
        """重置所有层状态"""
        self.bank_short.reset()
        self.bank_mid.reset()
        self.bank_long.reset()
        # Cycle 5: Reset ultra-long tier
        if self.use_ultra_long_tier and self.bank_ultra is not None:
            self.bank_ultra.reset()
        if self.quality_scorer is not None:
            self.quality_scorer.reset()
        if self.coherence_scorer is not None:
            self.coherence_scorer.reset()
        if self.progressive_activation is not None:
            self.progressive_activation.reset()
    
    def update(
        self,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        frame_hidden: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        更新所有层的float KV
        
        Args:
            evicted_k: 被驱逐的key [B, N, num_heads, head_dim]
            evicted_v: 被驱逐的value [B, N, num_heads, head_dim]
            frame_hidden: 当前帧的隐藏状态（用于质量评分/一致性，可选）
        
        Returns:
            stats: 各层更新统计信息
        """
        # 计算质量分数或一致性分数
        quality_score = 1.0
        coherence_scores = None
        coherence_overall = 1.0

        # Cycle 8: Progressive frame contribution scaling
        # Early frames get 30% quality weight, increasing to 100% over 30 updates
        # This prevents initial scene from dominating all slots (first-frame bias)
        self._update_call_count = getattr(self, '_update_call_count', 0) + 1
        warmup_scale = 0.3 + 0.7 * min(1.0, self._update_call_count / 30.0)

        if frame_hidden is not None:
            with torch.no_grad():
                if self.coherence_scorer is not None:
                    coherence_scores, coherence_overall = self.coherence_scorer(frame_hidden)
                    quality_score = coherence_overall
                    # Update coherence history
                    self.coherence_scorer.update_history(frame_hidden)
                elif self.quality_scorer is not None:
                    quality_score = self.quality_scorer(frame_hidden).mean().item()

        # Apply warmup scale to quality_score for progressive activation
        quality_score = quality_score * warmup_scale
        
        # 如果使用渐进式激活，调整各层的有效alpha
        if self.progressive_activation is not None:
            eff_alpha_short = self.progressive_activation.get_effective_alpha(
                self._alphas['short'], 'short')
            eff_alpha_mid = self.progressive_activation.get_effective_alpha(
                self._alphas['mid'], 'mid')
            eff_alpha_long = self.progressive_activation.get_effective_alpha(
                self._alphas['long'], 'long')
            # Temporarily override alpha values
            orig_alpha_short = self.bank_short.alpha
            orig_alpha_mid = self.bank_mid.alpha
            orig_alpha_long = self.bank_long.alpha
            self.bank_short.alpha = eff_alpha_short
            self.bank_mid.alpha = eff_alpha_mid
            self.bank_long.alpha = eff_alpha_long
        
        # 更新各层
        stats_short = self.bank_short.update(evicted_k, evicted_v, quality_score)
        stats_mid = self.bank_mid.update(evicted_k, evicted_v, quality_score)
        stats_long = self.bank_long.update(evicted_k, evicted_v, quality_score)
        # Cycle 5: Update ultra-long tier
        stats_ultra = {}
        if self.use_ultra_long_tier and self.bank_ultra is not None:
            stats_ultra = self.bank_ultra.update(evicted_k, evicted_v, quality_score)

        # 恢复原始alpha值
        if self.progressive_activation is not None:
            self.bank_short.alpha = orig_alpha_short
            self.bank_mid.alpha = orig_alpha_mid
            self.bank_long.alpha = orig_alpha_long

        # 构建返回统计信息
        result = {
            'short': stats_short,
            'mid': stats_mid,
            'long': stats_long,
            'ultra': stats_ultra,
            'quality_score': quality_score,
            'adaptive_alpha_short': stats_short.get('adaptive_alpha', self.bank_short.alpha),
            'adaptive_alpha_mid': stats_mid.get('adaptive_alpha', self.bank_mid.alpha),
            'adaptive_alpha_long': stats_long.get('adaptive_alpha', self.bank_long.alpha)
        }
        
        if coherence_scores is not None:
            result['coherence_scores'] = coherence_scores.cpu().tolist()
            result['coherence_overall'] = coherence_overall
        
        if self.progressive_activation is not None:
            result['progress'] = self.progressive_activation.get_progress()
            result['long_term_weight'] = self.progressive_activation.get_long_term_weight()
        
        return result
    
    def step(self, num_frames: int = 1):
        """
        推进时间步（用于渐进式激活）
        
        Args:
            num_frames: 推进的帧数
        """
        if self.progressive_activation is not None:
            self.progressive_activation.step(num_frames)
    
    def get_all_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取所有层的float K/V对，按slot维度拼接
        
        Returns:
            float_k: [total_slots, num_heads, head_dim]
            float_v: [total_slots, num_heads, head_dim]
        """
        k_short, v_short = self.bank_short.get_kv()
        k_mid, v_mid = self.bank_mid.get_kv()
        k_long, v_long = self.bank_long.get_kv()

        # 如果使用渐进式激活，根据权重调整长期bank的贡献
        if self.progressive_activation is not None and k_long.shape[0] > 0:
            long_weight = self.progressive_activation.get_long_term_weight()
            if long_weight < 1.0:
                v_long = v_long * long_weight

        # Cycle 5: Include ultra-long tier if active
        all_ks = [k_short, k_mid, k_long]
        all_vs = [v_short, v_mid, v_long]
        if self.use_ultra_long_tier and self.bank_ultra is not None:
            k_ultra, v_ultra = self.bank_ultra.get_kv()
            if k_ultra.shape[0] > 0:
                # Cycle 9: Remove the 0.5 V scaling on ultra-long tier
                # Prior: v_ultra * 0.5 combined with temporal decay → near-zero contribution
                # Now: rely only on temporal decay for natural downweighting of old context
                # This allows ultra-long tier to actually contribute to long-range consistency
                all_ks.append(k_ultra)
                all_vs.append(v_ultra)

        float_k = torch.cat(all_ks, dim=0)
        float_v = torch.cat(all_vs, dim=0)

        return float_k, float_v

    def is_ready(self) -> bool:
        '''Returns True if at least one bank has been initialized with at least one eviction.
        Used to guard against injecting zero-initialized float tokens.'''
        # Check which banks are active (num_slots > 0)
        if self.bank_short.num_slots > 0:
            return bool(self.bank_short.initialized.item())
        elif self.bank_mid.num_slots > 0:
            return bool(self.bank_mid.initialized.item())
        elif self.bank_long.num_slots > 0:
            return bool(self.bank_long.initialized.item())
        # Cycle 5: Check ultra-long tier as last resort
        elif self.use_ultra_long_tier and self.bank_ultra is not None and self.bank_ultra.num_slots > 0:
            return bool(self.bank_ultra.initialized.item())
        return False  # All banks disabled


# 便捷函数
def create_hierarchical_float_bank(
    d_model: int = 2048,
    num_slots: int = 4,
    alpha: float = 0.2,
    use_short: bool = True,
    use_mid: bool = True,
    use_long: bool = True,
    use_quality_scorer: bool = True
) -> Optional[HierarchicalFloatBank]:
    """
    便捷函数：创建分层 Float Bank

    默认配置：
    - 短期：每帧更新，alpha=0.3
    - 中期：每 30 帧更新，alpha=0.15
    - 长期：每 90 帧更新，alpha=0.05
    """
    if not any([use_short, use_mid, use_long]):
        return None

    return HierarchicalFloatBank(
        d_model=d_model,
        num_slots_short=num_slots if use_short else 0,
        num_slots_mid=num_slots if use_mid else 0,
        num_slots_long=num_slots if use_long else 0,
        alpha_short=0.3,
        alpha_mid=0.15,
        alpha_long=0.05,
        update_interval_short=1,
        update_interval_mid=30,
        update_interval_long=90,
        use_quality_scorer=use_quality_scorer
    )


# =============================================================================
# Query-Conditioned Slot Gating (QCSG) - Cycle 11 Improvement
# =============================================================================

class QueryConditionedSlotGating(nn.Module):
    """
    Query-Conditioned Slot Gating for Float KV Banks.

    Computes per-slot relevance scores based on query-key similarity,
    applies temporal decay, and uses soft gating to modulate contributions.

    Key improvements over Cycle 10:
    - Soft gating via softmax (vs. hard threshold)
    - Magnitude-aware K scaling (preserve relative magnitudes)
    - Unified temporal decay (applied to relevance, not just V)
    - Increased decay tau for 999-frame generation

    Args:
        head_dim: Dimension per attention head
        temperature: Softmax temperature for gating (default 0.5)
        decay_tau: Temporal decay time constant (default 150.0)
        min_gate_weight: Minimum gate weight to prevent complete suppression
    """

    def __init__(
        self,
        head_dim: int,
        temperature: float = 0.5,
        decay_tau: float = 150.0,
        min_gate_weight: float = 0.01,
        eps: float = 1e-6
    ):
        super().__init__()
        self.head_dim = head_dim
        self.temperature = temperature
        self.decay_tau = decay_tau
        self.min_gate_weight = min_gate_weight
        self.eps = eps

    def compute_relevance_scores(
        self,
        query: torch.Tensor,
        float_k: torch.Tensor,
        slot_staleness: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute query-conditioned relevance scores for each float slot.

        Args:
            query: [B, S_q, H, D] current queries
            float_k: [K, H, D] float token keys
            slot_staleness: [K] eviction steps since last update

        Returns:
            relevance: [B, K] relevance scores per slot
        """
        B, S_q, H, D = query.shape
        K = float_k.shape[0]

        # Compute query mean per head: [B, H, D]
        q_mean = query.mean(dim=1)  # average over sequence

        # Compute dot product similarity: [B, H, K]
        relevance_per_head = torch.einsum('bhd,khd->bhk', q_mean, float_k) / (D ** 0.5)

        # Average over heads: [B, K]
        relevance = relevance_per_head.mean(dim=1)

        # Apply temporal decay
        temporal_weight = torch.exp(-slot_staleness.float() / self.decay_tau)  # [K]
        relevance = relevance * temporal_weight.unsqueeze(0)  # [B, K]

        return relevance

    def compute_gate_weights(
        self,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft gating weights via temperature-scaled softmax.

        Args:
            relevance: [B, K] relevance scores

        Returns:
            gate_weights: [B, K] normalized gating weights
        """
        # Temperature-scaled softmax
        gate_weights = F.softmax(relevance / self.temperature, dim=-1)  # [B, K]

        # Ensure minimum contribution (prevent complete suppression)
        gate_weights = torch.clamp(gate_weights, min=self.min_gate_weight)

        # Renormalize after clamping
        gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + self.eps)

        return gate_weights

    def scale_float_k(
        self,
        float_k: torch.Tensor,
        cached_k_scale: float
    ) -> torch.Tensor:
        """
        Apply magnitude-aware scaling to float_k.

        Preserves relative magnitudes across slots, only applies global scale factor.

        Args:
            float_k: [K, H, D] float token keys
            cached_k_scale: scalar, mean norm of cached keys

        Returns:
            float_k_scaled: [K, H, D] scaled keys
        """
        float_k_norms = float_k.norm(dim=-1)  # [K, H]
        float_k_mean_norm = float_k_norms.mean().item()

        scale_factor = cached_k_scale / (float_k_mean_norm + self.eps)
        # Wider clamp than Cycle 10's 0.8-1.2 to allow better magnitude matching
        scale_factor = max(0.5, min(2.0, scale_factor))

        return float_k * scale_factor

    def forward(
        self,
        query: torch.Tensor,
        float_k: torch.Tensor,
        float_v: torch.Tensor,
        slot_staleness: torch.Tensor,
        cached_k: torch.Tensor
    ) -> tuple:
        """
        Apply query-conditioned gating to float KV slots.

        Args:
            query: [B, S_q, H, D]
            float_k: [K, H, D]
            float_v: [K, H, D]
            slot_staleness: [K]
            cached_k: [B, S_c, H, D]

        Returns:
            float_k_processed: [B, K, H, D] processed keys
            float_v_processed: [B, K, H, D] processed values
            gate_weights: [B, K] gating weights (for logging)
        """
        B = query.shape[0]

        # 1. Compute relevance scores
        relevance = self.compute_relevance_scores(query, float_k, slot_staleness)  # [B, K]

        # 2. Compute soft gating weights
        gate_weights = self.compute_gate_weights(relevance)  # [B, K]

        # 3. Scale float_k (magnitude-aware)
        cached_k_scale = cached_k.norm(dim=-1).mean().item()
        float_k_scaled = self.scale_float_k(float_k, cached_k_scale)  # [K, H, D]

        # 4. Apply gating to V values
        gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        float_v_gated = float_v.unsqueeze(0) * gate_weights_expanded  # [B, K, H, D]

        # 5. Expand float_k to batch dimension
        float_k_batch = float_k_scaled.unsqueeze(0).expand(B, -1, -1, -1)  # [B, K, H, D]

        return float_k_batch, float_v_gated, gate_weights


# =============================================================================
# Attention-Guided Float Tokens (AGFT) - Cycle 1 Implementation
# =============================================================================

class AttentionGuidedFloatBank(nn.Module):
    """
    Attention-Guided Float Tokens (AGFT) - A new approach to long-range consistency.
    
    Instead of injecting external KV pairs into attention (which contaminates subject 
    representations), AGFT computes guidance scores from float tokens and modulates
    attention weights within the local window.
    
    Key innovation: Multiplicative attention guidance rather than additive KV injection.
    
    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_slots_short/mid/long: Number of slots per temporal scale
        alpha_short/mid/long: EMA update coefficients
        guidance_alpha: Strength of attention modulation (default 0.1)
        temporal_weights: Weights for short/mid/long guidance aggregation
        use_guidance_dropout: Whether to apply dropout during training
        guidance_dropout_p: Dropout probability for training
    """
    
    def __init__(
        self,
        num_heads: int = 16,
        head_dim: int = 128,
        num_slots_short: int = 4,
        num_slots_mid: int = 4,
        num_slots_long: int = 4,
        alpha_short: float = 0.3,
        alpha_mid: float = 0.15,
        alpha_long: float = 0.05,
        update_interval_short: int = 1,
        update_interval_mid: int = 10,
        update_interval_long: int = 30,
        guidance_alpha: float = 0.1,
        temporal_weights: List[float] = None,
        use_guidance_dropout: bool = True,
        guidance_dropout_p: float = 0.1,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.guidance_alpha = guidance_alpha
        self.use_guidance_dropout = use_guidance_dropout
        self.guidance_dropout_p = guidance_dropout_p
        self.eps = eps
        
        # Temporal weights for aggregating short/mid/long guidance
        if temporal_weights is None:
            temporal_weights = [0.5, 0.3, 0.2]  # short, mid, long
        self.register_buffer('temporal_weights', torch.tensor(temporal_weights))
        
        # Hierarchical KV banks for storing compressed representations
        self.bank_short = FloatKVSlot(
            num_slots=num_slots_short,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_short,
            update_interval=update_interval_short,
            eps=eps,
            use_fifo_slots=True
        )
        
        self.bank_mid = FloatKVSlot(
            num_slots=num_slots_mid,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_mid,
            update_interval=update_interval_mid,
            eps=eps,
            use_fifo_slots=True
        )
        
        self.bank_long = FloatKVSlot(
            num_slots=num_slots_long,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_long,
            update_interval=update_interval_long,
            eps=eps,
            use_fifo_slots=True
        )
        
        # Learnable projection for query-to-float-token compatibility
        # This allows the model to adapt how queries match against stored tokens
        self.query_proj = nn.Linear(head_dim, head_dim, bias=False)
        
        # Initialize with small weights for stable start
        nn.init.normal_(self.query_proj.weight, std=0.02)
        
        self.total_slots = num_slots_short + num_slots_mid + num_slots_long
    
    def reset(self):
        """Reset all bank states."""
        self.bank_short.reset()
        self.bank_mid.reset()
        self.bank_long.reset()
    
    def update(
        self,
        evicted_k: torch.Tensor,
        evicted_v: torch.Tensor,
        frame_hidden: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Update all hierarchical banks with evicted KV pairs.
        
        Args:
            evicted_k: Evicted keys [B, N, num_heads, head_dim]
            evicted_v: Evicted values [B, N, num_heads, head_dim]
            frame_hidden: Current frame hidden state (optional)
        
        Returns:
            stats: Update statistics
        """
        stats_short = self.bank_short.update(evicted_k, evicted_v)
        stats_mid = self.bank_mid.update(evicted_k, evicted_v)
        stats_long = self.bank_long.update(evicted_k, evicted_v)
        
        return {
            'short': stats_short,
            'mid': stats_mid,
            'long': stats_long
        }
    
    def compute_guidance_scores(
        self,
        query: torch.Tensor,
        training: bool = False
    ) -> torch.Tensor:
        """
        Compute attention guidance scores from float tokens.
        
        This is the core AGFT operation. Instead of injecting K/V pairs, we compute
        similarity scores between queries and stored float tokens, then use these
        to create a guidance vector that modulates attention weights.
        
        Args:
            query: Query tensor [B, S, num_heads, head_dim] for current frame
            training: Whether in training mode (applies dropout if enabled)
            
        Returns:
            guidance: Guidance scores [B, S, num_heads] for attention modulation
        """
        B, S, H, D = query.shape
        
        # Project queries for better compatibility with stored tokens
        query_proj = self.query_proj(query)  # [B, S, H, D]
        
        # Get float tokens from all banks
        k_short, _ = self.bank_short.get_kv()  # [K_short, H, D]
        k_mid, _ = self.bank_mid.get_kv()      # [K_mid, H, D]
        k_long, _ = self.bank_long.get_kv()    # [K_long, H, D]
        
        # Compute guidance for each temporal scale
        guidance_list = []
        
        for bank_idx, (k_bank, weight) in enumerate(zip(
            [k_short, k_mid, k_long],
            self.temporal_weights
        )):
            if k_bank.shape[0] == 0:
                # No tokens in this bank yet
                guidance_list.append(torch.zeros(B, S, H, device=query.device, dtype=query.dtype))
                continue
            
            # Expand bank tokens to match batch and query dimensions
            # k_bank: [K, H, D] -> [B, H, K, D]
            k_expanded = k_bank.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)
            
            # query_proj: [B, S, H, D] -> [B, H, S, D]
            q_expanded = query_proj.transpose(1, 2)
            
            # Compute similarity: [B, H, S, K]
            scale = D ** -0.5
            similarity = torch.matmul(q_expanded, k_expanded.transpose(-2, -1)) * scale
            
            # Aggregate across tokens (max pooling for sharp selection)
            guidance_scale, _ = similarity.max(dim=-1)  # [B, H, S]
            guidance_scale = guidance_scale.transpose(1, 2)  # [B, S, H]
            
            # Apply temporal weight
            guidance_list.append(guidance_scale * weight)
        
        # Aggregate across temporal scales
        guidance = torch.stack(guidance_list, dim=-1).sum(dim=-1)  # [B, S, H]
        
        # Normalize to [0, 1] range using sigmoid
        guidance = torch.sigmoid(guidance)
        
        # Apply dropout during training for robustness
        if training and self.use_guidance_dropout and self.guidance_dropout_p > 0:
            guidance = F.dropout(guidance, p=self.guidance_dropout_p, training=True)
        
        return guidance
    
    def modulate_attention_weights(
        self,
        attention_logits: torch.Tensor,
        query: torch.Tensor,
        training: bool = False
    ) -> torch.Tensor:
        """
        Modulate attention logits using float token guidance.
        
        The modulation is multiplicative on the attention probabilities (post-softmax)
        rather than additive on logits, to preserve the causal structure.
        
        Args:
            attention_logits: Pre-softmax attention logits [B, H, S, T]
            query: Query tensor [B, S, H, D] for computing guidance
            training: Whether in training mode
            
        Returns:
            modulated_logits: Guided attention logits
        """
        # Compute guidance scores [B, S, H]
        guidance = self.compute_guidance_scores(query, training=training)
        
        # Expand guidance to match attention shape [B, H, S, 1]
        guidance_expanded = guidance.transpose(1, 2).unsqueeze(-1)
        
        # Modulate attention logits additively
        # The guidance boosts attention to positions that align with historical patterns
        # We add a bias proportional to the guidance score
        modulation = self.guidance_alpha * guidance_expanded
        
        # Apply modulation to the diagonal (self-attention within frame)
        # This encourages the frame to attend to positions consistent with history
        modulated_logits = attention_logits + modulation
        
        return modulated_logits
    
    def is_ready(self) -> bool:
        """Check if at least one bank has been initialized."""
        return (self.bank_short.initialized.item() or 
                self.bank_mid.initialized.item() or 
                self.bank_long.initialized.item())
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics from all banks."""
        return {
            'short': self.bank_short.get_stats() if hasattr(self.bank_short, 'get_stats') else {},
            'mid': self.bank_mid.get_stats() if hasattr(self.bank_mid, 'get_stats') else {},
            'long': self.bank_long.get_stats() if hasattr(self.bank_long, 'get_stats') else {},
            'guidance_alpha': self.guidance_alpha,
            'temporal_weights': self.temporal_weights.tolist()
        }


def create_agft_config(
    guidance_alpha: float = 0.1,
    temporal_weights: List[float] = None,
    num_slots_short: int = 4,
    num_slots_mid: int = 4,
    num_slots_long: int = 4,
    use_guidance_dropout: bool = True,
    guidance_dropout_p: float = 0.1
) -> dict:
    """
    Create configuration for Attention-Guided Float Tokens.

    Args:
        guidance_alpha: Strength of attention modulation (0.05-0.2 recommended)
        temporal_weights: Weights for short/mid/long guidance
        num_slots_short/mid/long: Number of slots per bank
        use_guidance_dropout: Enable dropout during training
        guidance_dropout_p: Dropout probability

    Returns:
        Configuration dict for model_kwargs
    """
    if temporal_weights is None:
        temporal_weights = [0.5, 0.3, 0.2]

    return {
        "use_float_tokens": True,
        "use_kv_bank_v2": False,  # AGFT replaces V2
        "use_attention_guided_float_tokens": True,
        "agft_guidance_alpha": guidance_alpha,
        "agft_temporal_weights": temporal_weights,
        "agft_num_slots_short": num_slots_short,
        "agft_num_slots_mid": num_slots_mid,
        "agft_num_slots_long": num_slots_long,
        "agft_use_guidance_dropout": use_guidance_dropout,
        "agft_guidance_dropout_p": guidance_dropout_p,
        "agft_update_interval_short": 1,
        "agft_update_interval_mid": 10,
        "agft_update_interval_long": 30,
    }


# =============================================================================
# 960-Frame Video Generation Enhancements
# =============================================================================

class PositionAdaptiveGuidance:
    """
    Adjusts guidance strength based on position in sequence for 960-frame videos.

    Schedule:
    - Frames 0-200: alpha = 0.03 (minimal guidance, scene establishment)
    - Frames 200-500: alpha = 0.08 (moderate guidance, scene development)
    - Frames 500-800: alpha = 0.12 (stronger guidance, stability)
    - Frames 800+: alpha = 0.15 (maximum guidance, prevent late drift)

    Uses smooth transitions between phases.
    """

    def __init__(self, base_alpha=0.1, max_frames=960):
        self.base_alpha = base_alpha
        self.max_frames = max_frames
        self.frame_count = 0

    def get_guidance_alpha(self, current_frame=None):
        """Get adaptive guidance alpha based on frame position."""
        if current_frame is None:
            current_frame = self.frame_count

        # Smooth piecewise linear with transitions
        progress = current_frame / self.max_frames

        if progress < 0.2:
            # Phase 1: Scene establishment (0-20%)
            return 0.03 + (0.05 * progress / 0.2)
        elif progress < 0.5:
            # Phase 2: Scene development (20-50%)
            return 0.08 + (0.04 * (progress - 0.2) / 0.3)
        elif progress < 0.8:
            # Phase 3: Stability (50-80%)
            return 0.12 + (0.03 * (progress - 0.5) / 0.3)
        else:
            # Phase 4: Late sequence (80-100%)
            return 0.15

    def step(self, num_frames=1):
        """Advance frame counter."""
        self.frame_count += num_frames

    def reset(self):
        """Reset frame counter."""
        self.frame_count = 0

    def get_progress(self) -> float:
        """Get current progress as ratio [0, 1]."""
        return min(1.0, self.frame_count / self.max_frames)


class SceneChangeDetector:
    """
    Detects scene transitions based on KV cache statistics.

    Detection criteria:
    1. Sudden change in average KV norm (>50% increase or <30% decrease)
    2. High variance in token norms within a frame (indicating mixed content)
    3. Coherence score dropping below threshold

    On detection: Reset appropriate float banks (short/mid only, preserve long)
    """

    def __init__(self,
                 norm_change_threshold=0.5,
                 coherence_drop_threshold=0.3,
                 history_size=10):
        self.norm_change_threshold = norm_change_threshold
        self.coherence_drop_threshold = coherence_drop_threshold
        self.history_size = history_size

        self.kv_norm_history = deque(maxlen=history_size)
        self.coherence_history = deque(maxlen=history_size)
        self.last_reset_frame = 0
        self.reset_cooldown = 30  # Minimum frames between resets

    def detect(self, current_kv_norm, coherence_score, current_frame):
        """
        Detect scene change.

        Returns:
            (scene_changed: bool, confidence: float)
        """
        self.kv_norm_history.append(current_kv_norm)
        self.coherence_history.append(coherence_score)

        # Cooldown period
        if current_frame - self.last_reset_frame < self.reset_cooldown:
            return False, 0.0

        if len(self.kv_norm_history) < 3:
            return False, 0.0

        # Check for norm spike/drop
        recent_norms = list(self.kv_norm_history)[-3:]
        avg_recent = sum(recent_norms[:-1]) / len(recent_norms[:-1])
        current = recent_norms[-1]

        norm_change = abs(current - avg_recent) / (avg_recent + 1e-6)
        norm_changed = norm_change > self.norm_change_threshold

        # Check for coherence drop
        if len(self.coherence_history) >= 3:
            recent_coherence = list(self.coherence_history)[-3:]
            avg_coherence = sum(recent_coherence[:-1]) / len(recent_coherence[:-1])
            coherence_drop = avg_coherence - recent_coherence[-1]
            coherence_changed = coherence_drop > self.coherence_drop_threshold
        else:
            coherence_changed = False
            coherence_drop = 0.0

        # Combined detection
        if norm_changed and coherence_changed:
            confidence = min(1.0, (norm_change + coherence_drop) / 2)
            self.last_reset_frame = current_frame
            return True, confidence
        elif norm_changed and len(self.kv_norm_history) >= 5:
            # Secondary check: sustained norm change
            older_avg = sum(list(self.kv_norm_history)[:-3]) / (len(self.kv_norm_history) - 3)
            if abs(current - older_avg) / (older_avg + 1e-6) > self.norm_change_threshold * 0.7:
                self.last_reset_frame = current_frame
                return True, norm_change

        return False, 0.0

    def reset(self):
        """Reset detector state."""
        self.kv_norm_history.clear()
        self.coherence_history.clear()
        self.last_reset_frame = 0


class ExtendedTemporalCoherenceScorer(TemporalCoherenceScorer):
    """
    Extended version with ultra-long window for 960-frame sequences.

    Windows:
    - Short: 3 frames (flicker detection)
    - Mid: 10 frames (local consistency)
    - Long: 30 frames (scene consistency)
    - Ultra: 100 frames (long-term drift detection)

    Additional: Trend analysis to detect gradual degradation.
    """

    def __init__(self, d_model, history_size=100):
        super().__init__(d_model, history_size)
        self.trend_buffer = deque(maxlen=20)  # For trend analysis
        self.degradation_counter = 0

    def compute_coherence_extended(self, current_frame):
        """
        Compute extended coherence with ultra-long window.

        Returns:
            (scores: [short, mid, long, ultra], overall, trend)
        """
        # Get base scores from parent (short, mid, long)
        base_scores, base_overall = super().compute_coherence(current_frame)

        # Compute ultra-long coherence (100 frames)
        if self.history_count >= 50:
            ultra_window = self._get_recent_frames(100)
            current_vec = current_frame.mean(dim=0) if current_frame.dim() == 2 else current_frame
            ultra_sim = F.cosine_similarity(
                current_vec.unsqueeze(0),
                ultra_window.mean(dim=0, keepdim=True),
                dim=-1
            ).mean()
        else:
            ultra_sim = torch.tensor(1.0, device=current_frame.device)

        # Stack all scores
        scores = torch.cat([base_scores, ultra_sim.unsqueeze(0)])

        # Compute trend (are we degrading over time?)
        self.trend_buffer.append(base_overall)
        trend = 0.0
        if len(self.trend_buffer) >= 10:
            recent = sum(list(self.trend_buffer)[-5:]) / 5
            older = sum(list(self.trend_buffer)[:5]) / 5
            trend = recent - older  # Negative = degrading

            # Track sustained degradation
            if trend < -0.05:
                self.degradation_counter += 1
            else:
                self.degradation_counter = max(0, self.degradation_counter - 1)

        # Adjust overall score based on trend and degradation
        adjusted_overall = base_overall
        if trend < -0.1:  # Significant degradation
            adjusted_overall = base_overall * 0.9
        if self.degradation_counter > 10:  # Sustained degradation
            adjusted_overall = adjusted_overall * 0.85

        return scores, adjusted_overall, trend


class EnhancedAttentionGuidedFloatBank(AttentionGuidedFloatBank):
    """
    Enhanced AGFT with position-adaptive guidance and scene change detection.
    Designed specifically for 960-frame video generation.
    """

    def __init__(self, *args,
                 use_position_adaptive=True,
                 use_scene_detection=True,
                 use_extended_coherence=True,
                 max_frames=960,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.use_position_adaptive = use_position_adaptive
        self.use_scene_detection = use_scene_detection
        self.use_extended_coherence = use_extended_coherence
        self.max_frames = max_frames

        if use_position_adaptive:
            self.position_guidance = PositionAdaptiveGuidance(
                base_alpha=self.guidance_alpha,
                max_frames=max_frames
            )

        if use_scene_detection:
            self.scene_detector = SceneChangeDetector()

        if use_extended_coherence:
            # Replace coherence scorer with extended version
            self.coherence_scorer = ExtendedTemporalCoherenceScorer(
                self.num_heads * self.head_dim,
                history_size=100
            )

        self.frame_counter = 0
        self.stats_history = []

    def compute_guidance_scores(self, query, training=False, current_frame=None):
        """
        Compute guidance scores with position-adaptive alpha.
        """
        # Get base guidance from parent
        guidance = super().compute_guidance_scores(query, training)

        # Apply position-adaptive scaling
        if self.use_position_adaptive and current_frame is not None:
            adaptive_alpha = self.position_guidance.get_guidance_alpha(current_frame)
            # Scale guidance by adaptive_alpha / base_alpha
            if self.guidance_alpha > 0:
                scale = adaptive_alpha / self.guidance_alpha
                guidance = guidance * scale

        return guidance

    def modulate_attention_weights(self, attention_logits, query, training=False, current_frame=None):
        """
        Modulate attention with position-adaptive guidance.
        """
        # Compute guidance with position adaptation
        guidance = self.compute_guidance_scores(query, training, current_frame)

        # Expand guidance to match attention shape [B, H, S, 1]
        guidance_expanded = guidance.transpose(1, 2).unsqueeze(-1)

        # Get adaptive alpha
        if self.use_position_adaptive and current_frame is not None:
            alpha = self.position_guidance.get_guidance_alpha(current_frame)
        else:
            alpha = self.guidance_alpha

        # Apply modulation
        modulation = alpha * guidance_expanded
        modulated_logits = attention_logits + modulation

        return modulated_logits

    def update(self, evicted_k, evicted_v, frame_hidden=None,
               current_frame=None, kv_norm=None):
        """
        Update with scene change detection.
        """
        # Update frame counter
        if current_frame is not None:
            self.frame_counter = current_frame
        else:
            current_frame = self.frame_counter
            self.frame_counter += 1

        # Check for scene change
        scene_changed = False
        confidence = 0.0
        if self.use_scene_detection and kv_norm is not None:
            # Get coherence if available
            coherence = 1.0
            if self.coherence_scorer is not None and frame_hidden is not None:
                if self.use_extended_coherence:
                    _, coherence, _ = self.coherence_scorer.compute_coherence_extended(frame_hidden)
                else:
                    _, coherence = self.coherence_scorer.compute_coherence(frame_hidden)

            scene_changed, confidence = self.scene_detector.detect(
                kv_norm, coherence, current_frame
            )

            if scene_changed and confidence > 0.6:
                # Reset short and mid banks, preserve long for context
                self.bank_short.reset()
                self.bank_mid.reset()
                print(f"[SceneChange] Detected at frame {current_frame} (confidence: {confidence:.2f})")

        # Normal update
        stats = super().update(evicted_k, evicted_v, frame_hidden)

        # Add scene detection info to stats
        stats['scene_changed'] = scene_changed
        stats['scene_confidence'] = confidence
        stats['frame'] = current_frame

        # Update position counter
        if self.use_position_adaptive:
            self.position_guidance.frame_count = current_frame

        self.stats_history.append(stats)
        return stats

    def reset(self):
        """Reset all components."""
        super().reset()
        if self.use_position_adaptive:
            self.position_guidance.reset()
        if self.use_scene_detection:
            self.scene_detector.reset()
        self.frame_counter = 0
        self.stats_history.clear()

    def get_detailed_stats(self):
        """Get comprehensive statistics."""
        base_stats = self.get_stats()
        base_stats['frame_counter'] = self.frame_counter
        base_stats['progress'] = self.frame_counter / self.max_frames if self.max_frames > 0 else 0

        if self.use_position_adaptive:
            base_stats['current_alpha'] = self.position_guidance.get_guidance_alpha()

        if self.stats_history:
            recent_scenes = [s for s in self.stats_history[-50:] if s.get('scene_changed')]
            base_stats['recent_scene_changes'] = len(recent_scenes)

        return base_stats


def create_960_frame_config(
    guidance_alpha: float = 0.1,
    temporal_weights: List[float] = None,
    num_slots_short: int = 4,
    num_slots_mid: int = 4,
    num_slots_long: int = 4,
    use_position_adaptive: bool = True,
    use_scene_detection: bool = True,
    use_extended_coherence: bool = True,
) -> dict:
    """
    Create optimized configuration for 960-frame video generation.

    Args:
        guidance_alpha: Base guidance strength
        temporal_weights: Weights for short/mid/long guidance
        num_slots_short/mid/long: Number of slots per bank
        use_position_adaptive: Enable position-adaptive guidance
        use_scene_detection: Enable scene change detection
        use_extended_coherence: Enable extended coherence scoring

    Returns:
        Configuration dict for model_kwargs
    """
    if temporal_weights is None:
        temporal_weights = [0.5, 0.3, 0.2]

    return {
        "use_float_tokens": True,
        "use_kv_bank_v2": False,
        "use_attention_guided_float_tokens": True,
        "use_enhanced_agft": True,
        "agft_guidance_alpha": guidance_alpha,
        "agft_temporal_weights": temporal_weights,
        "agft_num_slots_short": num_slots_short,
        "agft_num_slots_mid": num_slots_mid,
        "agft_num_slots_long": num_slots_long,
        "agft_update_interval_short": 1,
        "agft_update_interval_mid": 10,
        "agft_update_interval_long": 30,
        "agft_use_position_adaptive": use_position_adaptive,
        "agft_use_scene_detection": use_scene_detection,
        "agft_use_extended_coherence": use_extended_coherence,
        "agft_max_frames": 960,
    }
