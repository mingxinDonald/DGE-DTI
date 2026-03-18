"""Baseline MLP model for Drug-Target Interaction prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Simple MLP baseline that concatenates drug and target features directly.

    This model serves as a baseline against which more complex architectures
    (such as DGE-DTI) can be compared.

    Args:
        drug_input_dim: Dimensionality of drug feature vectors.
        target_input_dim: Dimensionality of target feature vectors.
        hidden_dim: Dimensionality of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        drug_input_dim: int,
        target_input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = drug_input_dim + target_input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, drug_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Run forward pass.

        Args:
            drug_features: Drug feature tensor of shape (batch, drug_input_dim).
            target_features: Target feature tensor of shape (batch, target_input_dim).

        Returns:
            Interaction probability tensor of shape (batch, 1).
        """
        combined = torch.cat([drug_features, target_features], dim=1)
        return self.network(combined)
