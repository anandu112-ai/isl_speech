"""
BiLSTM Temporal Landmark Classifier for INCLUDE-50 Sign Language Recognition.
Architectural flow:
Input (N, T, 225) -> Linear Projection + LayerNorm -> BiLSTM -> Temporal Mean & Max Pooling -> Classifier Head -> 50 Classes.
"""

import torch
import torch.nn as nn


class SignBiLSTMModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 225,
        proj_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 50,
    ):
        super().__init__()

        # Linear projection layer
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        bilstm_out_dim = hidden_size * 2  # 256

        # Classifier head with combined mean and max temporal pooling
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim * 2, hidden_size),  # 512 -> 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape: (batch_size, num_frames, feature_dim) = (N, 32, 225)
        Output shape: (batch_size, num_classes) = (N, 50)
        """
        proj = self.projection(x)  # (N, 32, 128)
        lstm_out, _ = self.bilstm(proj)  # (N, 32, 256)

        # Temporal Mean and Max Pooling
        mean_pool = torch.mean(lstm_out, dim=1)  # (N, 256)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (N, 256)

        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (N, 512)
        logits = self.classifier(pooled)  # (N, 50)

        return logits
