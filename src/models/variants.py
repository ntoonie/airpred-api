"""Four ablation variants, per thesis Table 1 / Data Generation Step 9.

Variant A: single-branch TCN, unified encoding (PM2.5 + met concatenated
           BEFORE encoding), concatenation fusion (trivial, one stream).
Variant B: dual-branch TCN, modality-specific encoding, concatenation fusion.
Variant C: dual-branch TCN, modality-specific encoding, cross-modal
           attention fusion. This is AIRPRED.
Variant D: single-branch TCN, PM2.5-only input, no meteorology at all.

Everything not listed above (TCN block design, forecast head design,
training hyperparameters) is IDENTICAL across all four, by design --
see implementation guide Section 24 (fair-comparison checklist).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tcn import TCNEncoder
from src.models.attention import CrossModalAttentionFusion


class ForecastHead(nn.Module):
    """Shared by all four variants (thesis Table 6)."""

    def __init__(self, in_dim: int, hidden: int = 128, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, in_dim)
        x = x.mean(dim=1)             # Global Average Pooling over L -> (B, in_dim)
        x = self.act(self.fc1(x))     # (B, hidden)
        x = self.drop(x)
        return self.fc2(x)            # (B, horizon)


class VariantA_SingleBranchUnified(nn.Module):
    """RQ1 baseline."""

    def __init__(self, n_met_features: int = 7, horizon: int = 24):
        super().__init__()
        self.encoder = TCNEncoder(in_channels=1 + n_met_features)
        self.head = ForecastHead(in_dim=128, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_pm25, x_met], dim=-1)
        h = self.encoder(x)
        return self.head(h)


class VariantB_DualBranchConcat(nn.Module):
    """RQ1 proposed model / RQ2 baseline."""

    def __init__(self, n_met_features: int = 7, horizon: int = 24):
        super().__init__()
        self.pm25_encoder = TCNEncoder(in_channels=1)
        self.met_encoder = TCNEncoder(in_channels=n_met_features)
        self.head = ForecastHead(in_dim=256, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor) -> torch.Tensor:
        h_pm25 = self.pm25_encoder(x_pm25)
        h_met = self.met_encoder(x_met)
        fused = torch.cat([h_pm25, h_met], dim=-1)
        return self.head(fused)


class VariantC_AIRPRED(nn.Module):
    """RQ2 & RQ3 proposed model. This is AIRPRED."""

    def __init__(self, n_met_features: int = 7, horizon: int = 24,
                 d_model: int = 128, num_heads: int = 4):
        super().__init__()
        self.pm25_encoder = TCNEncoder(in_channels=1)
        self.met_encoder = TCNEncoder(in_channels=n_met_features)
        self.fusion = CrossModalAttentionFusion(
            branch_dim=128, d_model=d_model, num_heads=num_heads, fused_dim=256
        )
        self.head = ForecastHead(in_dim=256, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor, return_attention: bool = False):
        h_pm25 = self.pm25_encoder(x_pm25)
        h_met = self.met_encoder(x_met)
        fused, attn_weights = self.fusion(h_pm25, h_met)
        out = self.head(fused)
        return (out, attn_weights) if return_attention else out


class VariantD_PM25Only(nn.Module):
    """RQ3 baseline. x_met accepted-and-ignored for a uniform call signature
    across all four variants (simplifies the shared training loop)."""

    def __init__(self, horizon: int = 24):
        super().__init__()
        self.encoder = TCNEncoder(in_channels=1)
        self.head = ForecastHead(in_dim=128, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x_pm25)
        return self.head(h)


if __name__ == "__main__":
    B, L, H = 4, 48, 24
    x_pm25, x_met = torch.randn(B, L, 1), torch.randn(B, L, 7)
    for Model in (
        VariantA_SingleBranchUnified,
        VariantB_DualBranchConcat,
        VariantC_AIRPRED,
        VariantD_PM25Only,
    ):
        model = Model(horizon=H)
        out = model(x_pm25, x_met)
        assert out.shape == (B, H), f"{Model.__name__} shape mismatch: {out.shape}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{Model.__name__}: output {tuple(out.shape)}, {n_params:,} parameters")
    print("All four variants: forward pass + shape checks passed.")
