"""Preprocessing utilities for DTI data."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_smiles(smiles: str) -> bool:
    """Check whether a SMILES string can be parsed by RDKit.

    If RDKit is not available, returns True to allow processing to continue.

    Args:
        smiles: SMILES string to validate.

    Returns:
        True if the SMILES is valid, False otherwise.
    """
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return True


def validate_sequence(sequence: str, min_length: int = 5) -> bool:
    """Check whether a protein sequence meets basic quality criteria.

    Args:
        sequence: Amino acid sequence to validate.
        min_length: Minimum acceptable sequence length.

    Returns:
        True if the sequence is valid.
    """
    if len(sequence) < min_length:
        return False
    valid_fraction = sum(aa in VALID_AMINO_ACIDS for aa in sequence.upper()) / len(sequence)
    return valid_fraction >= 0.9


def filter_dataset(
    df: pd.DataFrame,
    drug_smiles_col: str = "drug_smiles",
    target_sequence_col: str = "target_sequence",
) -> pd.DataFrame:
    """Remove rows with invalid drug SMILES or protein sequences.

    Args:
        df: Input dataframe.
        drug_smiles_col: Column containing SMILES strings.
        target_sequence_col: Column containing protein sequences.

    Returns:
        Filtered dataframe.
    """
    initial_size = len(df)
    mask = df[drug_smiles_col].apply(lambda s: validate_smiles(str(s))) & df[
        target_sequence_col
    ].apply(lambda s: validate_sequence(str(s)))
    df_filtered = df[mask].reset_index(drop=True)
    removed = initial_size - len(df_filtered)
    if removed:
        logger.warning("Removed %d invalid rows during filtering.", removed)
    return df_filtered


def split_indices(
    n_samples: int,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split sample indices into train, validation, and test sets.

    Args:
        n_samples: Total number of samples.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    all_indices = list(range(n_samples))
    train_val, test_idx = train_test_split(
        all_indices, test_size=test_split, random_state=seed
    )
    adjusted_val = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val, test_size=adjusted_val, random_state=seed
    )
    return train_idx, val_idx, test_idx


def normalise_labels(labels: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalise continuous labels to [0, 1] range.

    Args:
        labels: 1-D array of continuous label values.

    Returns:
        Tuple of (normalised_labels, min_val, max_val).
    """
    min_val = float(labels.min())
    max_val = float(labels.max())
    if max_val - min_val < 1e-9:
        return np.zeros_like(labels, dtype=np.float32), min_val, max_val
    normalised = (labels - min_val) / (max_val - min_val)
    return normalised.astype(np.float32), min_val, max_val
