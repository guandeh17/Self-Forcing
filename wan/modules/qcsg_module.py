"""
Query-Conditioned Slot Gating (QCSG) - Cycle 11 Improvement

Implements soft gating mechanism for float token injection based on
query-key similarity and temporal decay.

Author: Research Agent
Date: 2026-04-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # q_mean: [B, H, D], float_k: [K, H, D]
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
        # Compute mean norm of float_k
        float_k_norms = float_k.norm(dim=-1)  # [K, H]
        float_k_mean_norm = float_k_norms.mean().item()

        # Compute global scale factor
        scale_factor = cached_k_scale / (float_k_mean_norm + self.eps)

        # Clamp to reasonable range (wider than Cycle 10's 0.8-1.2)
        scale_factor = max(0.5, min(2.0, scale_factor))

        # Apply uniform scaling (preserves relative magnitudes)
        float_k_scaled = float_k * scale_factor

        return float_k_scaled

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
        K = float_k.shape[0]

        # 1. Compute relevance scores
        relevance = self.compute_relevance_scores(query, float_k, slot_staleness)  # [B, K]

        # 2. Compute soft gating weights
        gate_weights = self.compute_gate_weights(relevance)  # [B, K]

        # 3. Scale float_k (magnitude-aware)
        cached_k_scale = cached_k.norm(dim=-1).mean().item()
        float_k_scaled = self.scale_float_k(float_k, cached_k_scale)  # [K, H, D]

        # 4. Apply gating to V values
        # gate_weights: [B, K] -> [B, K, 1, 1]
        gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)
        float_v_gated = float_v.unsqueeze(0) * gate_weights_expanded  # [B, K, H, D]

        # 5. Expand float_k to batch dimension
        float_k_batch = float_k_scaled.unsqueeze(0).expand(B, -1, -1, -1)  # [B, K, H, D]

        return float_k_batch, float_v_gated, gate_weights
