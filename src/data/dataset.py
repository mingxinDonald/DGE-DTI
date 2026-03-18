"""PyTorch Dataset implementation for Drug-Target Interaction data."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features.drug_features import extract_drug_features, get_drug_feature_dim
from features.target_features import extract_target_features, get_target_feature_dim

logger = logging.getLogger(__name__)


class DTIDataset(Dataset):
    """Dataset for drug-target interaction prediction.

    Loads drug SMILES and protein sequences from a CSV file, computes feature
    vectors, and stores labels for supervised learning.

    Args:
        data_path: Path to a CSV file with drug and target information.
        drug_smiles_col: Column name for drug SMILES strings.
        target_sequence_col: Column name for protein sequences.
        label_col: Column name for binary interaction labels.
        morgan_radius: Morgan fingerprint radius.
        morgan_bits: Number of Morgan fingerprint bits.
        use_maccs: Whether to include MACCS keys.
        max_seq_length: Maximum protein sequence length for normalisation.
        use_amino_acid_composition: Whether to include amino acid composition.
        indices: Optional subset of row indices to include.
    """

    def __init__(
        self,
        data_path: str,
        drug_smiles_col: str = "drug_smiles",
        target_sequence_col: str = "target_sequence",
        label_col: str = "label",
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        use_maccs: bool = True,
        max_seq_length: int = 1000,
        use_amino_acid_composition: bool = True,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.drug_smiles_col = drug_smiles_col
        self.target_sequence_col = target_sequence_col
        self.label_col = label_col
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.use_maccs = use_maccs
        self.max_seq_length = max_seq_length
        self.use_amino_acid_composition = use_amino_acid_composition

        df = pd.read_csv(data_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.drug_features: List[np.ndarray] = []
        self.target_features: List[np.ndarray] = []
        self.labels: List[float] = []

        logger.info("Processing %d samples from %s", len(df), data_path)
        skipped = 0
        for _, row in df.iterrows():
            smiles = str(row[drug_smiles_col])
            sequence = str(row[target_sequence_col])
            label = float(row[label_col])

            d_feat = extract_drug_features(
                smiles,
                morgan_radius=morgan_radius,
                morgan_bits=morgan_bits,
                use_maccs=use_maccs,
            )
            t_feat = extract_target_features(
                sequence,
                max_seq_length=max_seq_length,
                use_amino_acid_composition=use_amino_acid_composition,
            )

            if d_feat is None or t_feat is None:
                skipped += 1
                continue

            self.drug_features.append(d_feat)
            self.target_features.append(t_feat)
            self.labels.append(label)

        if skipped:
            logger.warning("Skipped %d samples due to parsing errors.", skipped)
        logger.info("Loaded %d valid samples.", len(self.labels))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "drug_features": torch.tensor(self.drug_features[idx], dtype=torch.float32),
            "target_features": torch.tensor(self.target_features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

    @property
    def drug_feature_dim(self) -> int:
        """Dimensionality of drug features."""
        return get_drug_feature_dim(self.morgan_bits, self.use_maccs)

    @property
    def target_feature_dim(self) -> int:
        """Dimensionality of target features."""
        return get_target_feature_dim(self.use_amino_acid_composition)

    def get_labels(self) -> List[float]:
        """Return all labels as a Python list."""
        return list(self.labels)
