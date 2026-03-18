"""DGE-DTI: Drug Graph Encoder for Drug-Target Interaction prediction.

This module implements the main DGE-DTI model, which uses separate encoders
for drugs and protein targets before combining their representations for
interaction prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugEncoder(nn.Module):
    """Multi-layer perceptron encoder for drug feature vectors.

    Encodes molecular fingerprints into a dense latent representation.

    Args:
        input_dim: Dimensionality of the input drug feature vector.
        hidden_dim: Dimensionality of hidden layers.
        output_dim: Dimensionality of the output embedding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode drug features.

        Args:
            x: Drug feature tensor of shape (batch, input_dim).

        Returns:
            Drug embedding tensor of shape (batch, output_dim).
        """
        return self.network(x)


class TargetEncoder(nn.Module):
    """Multi-layer perceptron encoder for protein target feature vectors.

    Encodes sequence-derived features into a dense latent representation.

    Args:
        input_dim: Dimensionality of the input target feature vector.
        hidden_dim: Dimensionality of hidden layers.
        output_dim: Dimensionality of the output embedding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode target features.

        Args:
            x: Target feature tensor of shape (batch, input_dim).

        Returns:
            Target embedding tensor of shape (batch, output_dim).
        """
        return self.network(x)


class InteractionPredictor(nn.Module):
    """Interaction predictor that combines drug and target embeddings.

    Takes concatenated drug and target embeddings and outputs an interaction
    probability via a feed-forward network.

    Args:
        drug_dim: Dimensionality of drug embeddings.
        target_dim: Dimensionality of target embeddings.
        hidden_dim: Dimensionality of hidden layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        drug_dim: int = 128,
        target_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = drug_dim + target_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, drug_embedding: torch.Tensor, target_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Predict interaction probability.

        Args:
            drug_embedding: Drug embedding of shape (batch, drug_dim).
            target_embedding: Target embedding of shape (batch, target_dim).

        Returns:
            Interaction probability tensor of shape (batch, 1).
        """
        combined = torch.cat([drug_embedding, target_embedding], dim=1)
        return self.network(combined)


class DGEDTI(nn.Module):
    """Drug Graph Encoder for Drug-Target Interaction (DGE-DTI) model.

    End-to-end model that encodes drug and target features independently and
    predicts their interaction probability.

    Args:
        drug_input_dim: Dimensionality of drug feature vectors.
        target_input_dim: Dimensionality of target feature vectors.
        drug_hidden_dim: Hidden dimension for drug encoder.
        target_hidden_dim: Hidden dimension for target encoder.
        embedding_dim: Shared embedding dimensionality for drug and target.
        interaction_hidden_dim: Hidden dimension for interaction predictor.
        dropout: Dropout probability applied throughout.
    """

    def __init__(
        self,
        drug_input_dim: int,
        target_input_dim: int,
        drug_hidden_dim: int = 256,
        target_hidden_dim: int = 256,
        embedding_dim: int = 128,
        interaction_hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.drug_encoder = DrugEncoder(
            input_dim=drug_input_dim,
            hidden_dim=drug_hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.target_encoder = TargetEncoder(
            input_dim=target_input_dim,
            hidden_dim=target_hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )
        self.interaction_predictor = InteractionPredictor(
            drug_dim=embedding_dim,
            target_dim=embedding_dim,
            hidden_dim=interaction_hidden_dim,
            dropout=dropout,
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
        drug_emb = self.drug_encoder(drug_features)
        target_emb = self.target_encoder(target_features)
        return self.interaction_predictor(drug_emb, target_emb)

    def encode_drug(self, drug_features: torch.Tensor) -> torch.Tensor:
        """Encode only the drug features.

        Args:
            drug_features: Drug feature tensor of shape (batch, drug_input_dim).

        Returns:
            Drug embedding tensor of shape (batch, embedding_dim).
        """
        return self.drug_encoder(drug_features)

    def encode_target(self, target_features: torch.Tensor) -> torch.Tensor:
        """Encode only the target features.

        Args:
            target_features: Target feature tensor of shape (batch, target_input_dim).

        Returns:
            Target embedding tensor of shape (batch, embedding_dim).
        """
        return self.target_encoder(target_features)
