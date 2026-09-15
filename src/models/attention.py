"""Cross-modal attention fusion module, per thesis Table 5.

PM2.5 branch output = source of Query.
Meteorological branch output = source of Key and Value.

NOTE: `d_model` is NOT assigned a numeric value anywhere in the thesis
(Table 5 uses the symbol only). The default of 128 here is a documented
ASSUMPTION -- see implementation guide Section 16/19 -- not a thesis fact.
Confirm with your adviser before treating this as final.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    def __init__(
        self,
        branch_dim: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        fused_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # nn.MultiheadAttention's query path always requires its input dim to
        # equal embed_dim -- kdim/vdim can differ from embed_dim, but there's
        # no separate "qdim". So branch_dim must equal d_model for h_pm25 to
        # be usable directly as the query below. If these ever need to
        # differ, add an explicit query-only projection before self.mha
        # (and remove this assert) -- do NOT add separate K/V projections
        # too, since nn.MultiheadAttention already provides those internally.
        assert branch_dim == d_model, (
            f"branch_dim ({branch_dim}) must equal d_model ({d_model}) -- "
            f"see comment above for why, and what to do if this changes."
        )
        # nn.MultiheadAttention already applies its own internal Q/K/V linear
        # projections (in_proj_weight) before computing attention -- that IS
        # the "Query/Key/Value Projection (Linear)" step from thesis Table 5.
        # Passing raw h_pm25 / h_met directly (not pre-projected) is correct
        # usage. The previous version of this file applied separate
        # q_proj/k_proj/v_proj layers in front of this -- stacking a second
        # linear transform with no nonlinearity in between adds ~49.5K
        # parameters with zero additional representational power (two
        # stacked linear layers collapse to one, algebraically), which likely
        # made this module modestly harder to optimize within the same fixed
        # epoch budget every other variant gets, for no benefit.
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.residual_proj = (
            nn.Identity() if branch_dim == d_model else nn.Linear(branch_dim, d_model)
        )
        self.output_proj = nn.Linear(d_model, fused_dim)

    def forward(self, h_pm25: torch.Tensor, h_met: torch.Tensor):
        """h_pm25, h_met: (B, L, branch_dim) -> fused (B, L, fused_dim), attn (B, h, L, L)."""
        attn_out, attn_weights = self.mha(
            h_pm25, h_met, h_met, need_weights=True, average_attn_weights=False
        )
        z = self.norm1(attn_out + self.residual_proj(h_pm25))
        z = self.norm2(self.ffn(z) + z)
        fused = self.output_proj(z)
        return fused, attn_weights


if __name__ == "__main__":
    fusion = CrossModalAttentionFusion()
    h_pm25 = torch.randn(4, 48, 128)
    h_met = torch.randn(4, 48, 128)
    fused, w = fusion(h_pm25, h_met)
    assert fused.shape == (4, 48, 256)
    print("Cross-modal attention shape check passed:", fused.shape, w.shape)