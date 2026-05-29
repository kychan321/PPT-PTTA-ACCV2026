import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    """
    A lightweight residual adapter for P-TTA.

    Input shape:
        [B, T, D]

    Output shape:
        [B, T, D]

    This module is inserted after PPT's trajectory encoder:
        past_feat = traj_encoder(past)
        past_feat = past_adapter(past_feat)

    The final linear layer is zero-initialized so that, at initialization,
    the adapter behaves like an identity function:
        output ≈ input
    """

    def __init__(self, dim=128, bottleneck=32, dropout=0.0):
        super().__init__()

        self.dim = dim
        self.bottleneck = bottleneck

        layers = [
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(bottleneck, dim))

        self.net = nn.Sequential(*layers)

        # Zero initialization:
        # At the beginning, self.net(x) ≈ 0, so forward(x) ≈ x.
        # This prevents the adapter from suddenly damaging the pretrained PPT behavior.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)