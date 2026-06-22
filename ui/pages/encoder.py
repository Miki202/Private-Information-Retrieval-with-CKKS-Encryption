"""
Копие на Encoder класа за UI
Копирай това от твоя AE.ipynb notebook
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder от Autoencoder модела"""
    def __init__(self, latent_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 256 → 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 128 → 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 64 → 32
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 32 * 32, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z