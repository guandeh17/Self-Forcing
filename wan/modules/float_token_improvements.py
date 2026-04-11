"""
Float Token 算法改进模块

包含改进的 Float Token 实现：
- FloatTokenBank: EMA 动态更新机制
- FrameQualityScorer: 帧质量感知筛选
- HierarchicalFloatBank: 分层 Float Token 设计
- RoPE 对齐辅助函数

作者: Claude
日期: 2026-04-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple


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

        # 4. 综合评分
        # 相似度过低（<0.3）：可能是崩坏或场景切换
        # 相似度过高（>0.95）：可能是静态或重复
        # 方差异常：可能是模糊或噪声

        # 理想情况：中等相似度 + 正常方差
        sim_quality = 1.0 - torch.abs(similarity - 0.6) * 2.0  # 峰值在 0.6
        sim_quality = torch.clamp(sim_quality, 0.0, 1.0)

        # 方差应该在一个合理范围内（不太低=不模糊，不太高=不噪声）
        var_norm = torch.clamp(variance_feature / (variance_feature + 0.1), 0.0, 1.0)
        var_quality = 1.0 - torch.abs(var_norm - 0.5) * 2.0  # 峰值在 0.5
        var_quality = torch.clamp(var_quality, 0.0, 1.0)

        # 组合质量分数
        quality = 0.6 * sim_quality + 0.4 * var_quality

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
    """
    
    def __init__(
        self,
        num_slots: int = 4,
        num_heads: int = 16,
        head_dim: int = 128,
        alpha: float = 0.2,
        update_interval: int = 1,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.alpha = alpha
        self.update_interval = update_interval
        self.eps = eps
        
        # 直接存储K/V对 - shape: [num_slots, num_heads, head_dim]
        self.register_buffer('slots_k', torch.zeros(num_slots, num_heads, head_dim))
        self.register_buffer('slots_v', torch.zeros(num_slots, num_heads, head_dim))
        
        # 状态标记
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
        
        # Content-Adaptive Alpha: 存储前一次的evicted_k平均值
        self.register_buffer('prev_evicted_k_avg', torch.zeros(num_heads, head_dim))
        self.register_buffer('has_prev', torch.tensor(False, dtype=torch.bool))
    
    def reset(self):
        """重置states状态"""
        self.slots_k.zero_()
        self.slots_v.zero_()
        self.initialized.fill_(False)
        self.update_count.zero_()
        self.prev_evicted_k_avg.zero_()
        self.has_prev.fill_(False)
    
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
            
            # 限制范围: [alpha_base * 0.5, min(alpha_base * 3, 0.9)]
            alpha_min = self.alpha * 0.5
            alpha_max = min(self.alpha * 3.0, 0.9)
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
        self.update_count += 1
        
        # 检查更新间隔
        if self.update_count % self.update_interval != 0:
            return {
                'updated': False,
                'update_count': self.update_count.item(),
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
                return {
                    'updated': True,
                    'initialized': True,
                    'slot_k_norms': self.slots_k.norm(dim=-1).mean(dim=-1).detach()
                }
            
            # Slot Attention: 在head维度上平均后计算attention
            # slots_k: [K, H, D] -> avg over heads -> [K, D]
            # ek: [N, H, D] -> avg over heads -> [N, D]
            slots_k_avg = self.slots_k.mean(dim=1)  # [K, D]
            ek_avg = ek.mean(dim=1)  # [N, D]
            ev_avg = ev.mean(dim=1)  # [N, D]
            
            # 计算attention logits: [K, N]
            attn_logits = torch.einsum('kd,nd->kn', slots_k_avg, ek_avg) * (self.head_dim ** -0.5)
            attn_weights = F.softmax(attn_logits, dim=-1)  # [K, N] - 每个slot对所有evicted tokens的权重
            
            # 计算新的K/V: 用attention weights加权求和
            # [K, N] @ [N, H, D] -> [K, H, D]
            new_k = torch.einsum('kn,nhd->khd', attn_weights, ek)
            new_v = torch.einsum('kn,nhd->khd', attn_weights, ev)
            
            # 计算内容自适应alpha
            adaptive_alpha = self._compute_adaptive_alpha(ek)
            
            # 结合质量分数计算有效alpha
            eff_alpha = max(0.05, min(0.95, adaptive_alpha * quality_score))
            
            # EMA更新
            self.slots_k.mul_(1 - eff_alpha).add_(new_k, alpha=eff_alpha)
            self.slots_v.mul_(1 - eff_alpha).add_(new_v, alpha=eff_alpha)
        
        return {
            'updated': True,
            'initialized': True,
            'alpha': eff_alpha,
            'adaptive_alpha': adaptive_alpha,
            'quality_score': quality_score,
            'slot_k_norms': self.slots_k.norm(dim=-1).mean(dim=-1).detach()
        }
    
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前所有slot的K/V对
        
        Returns:
            slots_k: [num_slots, num_heads, head_dim]
            slots_v: [num_slots, num_heads, head_dim]
        """
        return self.slots_k, self.slots_v


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
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_quality_scorer = use_quality_scorer
        
        # 创建三层FloatKVSlot
        self.bank_short = FloatKVSlot(
            num_slots=num_slots_short,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_short,
            update_interval=update_interval_short,
            eps=eps
        )
        
        self.bank_mid = FloatKVSlot(
            num_slots=num_slots_mid,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_mid,
            update_interval=update_interval_mid,
            eps=eps
        )
        
        self.bank_long = FloatKVSlot(
            num_slots=num_slots_long,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha_long,
            update_interval=update_interval_long,
            eps=eps
        )
        
        # 质量评分器
        if use_quality_scorer:
            self.quality_scorer = FrameQualityScorer(num_heads * head_dim, eps)
        else:
            self.quality_scorer = None
        
        self.total_slots = num_slots_short + num_slots_mid + num_slots_long
    
    def reset(self):
        """重置所有层状态"""
        self.bank_short.reset()
        self.bank_mid.reset()
        self.bank_long.reset()
        if self.quality_scorer is not None:
            self.quality_scorer.reset()
    
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
            frame_hidden: 当前帧的隐藏状态（用于质量评分，可选）
        
        Returns:
            stats: 各层更新统计信息
        """
        # 计算质量分数
        quality_score = 1.0
        if self.quality_scorer is not None and frame_hidden is not None:
            with torch.no_grad():
                quality_score = self.quality_scorer(frame_hidden).mean().item()
        
        # 更新各层
        stats_short = self.bank_short.update(evicted_k, evicted_v, quality_score)
        stats_mid = self.bank_mid.update(evicted_k, evicted_v, quality_score)
        stats_long = self.bank_long.update(evicted_k, evicted_v, quality_score)
        
        return {
            'short': stats_short,
            'mid': stats_mid,
            'long': stats_long,
            'quality_score': quality_score,
            'adaptive_alpha_short': stats_short.get('adaptive_alpha', self.bank_short.alpha),
            'adaptive_alpha_mid': stats_mid.get('adaptive_alpha', self.bank_mid.alpha),
            'adaptive_alpha_long': stats_long.get('adaptive_alpha', self.bank_long.alpha)
        }
    
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
        
        float_k = torch.cat([k_short, k_mid, k_long], dim=0)
        float_v = torch.cat([v_short, v_mid, v_long], dim=0)
        
        return float_k, float_v

    def is_ready(self) -> bool:
        '''Returns True if float bank has been initialized with at least one eviction.
        Used to guard against injecting zero-initialized float tokens.'''
        return bool(self.bank_short.initialized.item())


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
