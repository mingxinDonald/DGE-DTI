"""Protein/target feature extraction from amino acid sequences."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Physicochemical properties (hydrophobicity, charge, polarity, molecular weight)
# Values adapted from standard biochemistry references.
_AA_PROPERTIES: Dict[str, List[float]] = {
    "A": [1.8, 0.0, 0.0, 89.1],
    "C": [2.5, 0.0, 0.0, 121.2],
    "D": [-3.5, -1.0, 1.0, 133.1],
    "E": [-3.5, -1.0, 1.0, 147.1],
    "F": [2.8, 0.0, 0.0, 165.2],
    "G": [-0.4, 0.0, 0.0, 75.1],
    "H": [-3.2, 0.1, 1.0, 155.2],
    "I": [4.5, 0.0, 0.0, 131.2],
    "K": [-3.9, 1.0, 1.0, 146.2],
    "L": [3.8, 0.0, 0.0, 131.2],
    "M": [1.9, 0.0, 0.0, 149.2],
    "N": [-3.5, 0.0, 1.0, 132.1],
    "P": [-1.6, 0.0, 0.0, 115.1],
    "Q": [-3.5, 0.0, 1.0, 146.2],
    "R": [-4.5, 1.0, 1.0, 174.2],
    "S": [-0.8, 0.0, 1.0, 105.1],
    "T": [-0.7, 0.0, 1.0, 119.1],
    "V": [4.2, 0.0, 0.0, 117.1],
    "W": [-0.9, 0.0, 0.0, 204.2],
    "Y": [-1.3, 0.0, 1.0, 181.2],
}


def compute_amino_acid_composition(sequence: str) -> np.ndarray:
    """Compute amino acid composition (AAC) feature vector.

    Each element represents the fraction of the corresponding amino acid in the
    sequence, yielding a 20-dimensional vector.

    Args:
        sequence: Protein sequence string (single-letter codes).

    Returns:
        Numpy array of shape (20,) with relative amino acid frequencies.
    """
    sequence = sequence.upper()
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    valid = 0
    for aa in sequence:
        if aa in _AA_INDEX:
            counts[_AA_INDEX[aa]] += 1
            valid += 1
    if valid > 0:
        counts /= valid
    return counts


def compute_physicochemical_features(sequence: str) -> np.ndarray:
    """Compute mean physicochemical property vector for a protein sequence.

    Returns the mean of 4 physicochemical properties across all residues,
    yielding a 4-dimensional vector.

    Args:
        sequence: Protein sequence string (single-letter codes).

    Returns:
        Numpy array of shape (4,) with mean physicochemical properties.
    """
    sequence = sequence.upper()
    props: List[List[float]] = []
    for aa in sequence:
        if aa in _AA_PROPERTIES:
            props.append(_AA_PROPERTIES[aa])
    if not props:
        return np.zeros(4, dtype=np.float32)
    return np.mean(np.array(props, dtype=np.float32), axis=0)


def compute_sequence_length_features(sequence: str, max_length: int = 1000) -> np.ndarray:
    """Encode normalised sequence length and a length-bucket flag.

    Args:
        sequence: Protein sequence string.
        max_length: Maximum sequence length used for normalisation.

    Returns:
        Numpy array of shape (2,): [normalised_length, is_long_protein].
    """
    length = len(sequence)
    normalised = min(length / max_length, 1.0)
    is_long = float(length > max_length // 2)
    return np.array([normalised, is_long], dtype=np.float32)


def extract_target_features(
    sequence: str,
    max_seq_length: int = 1000,
    use_amino_acid_composition: bool = True,
) -> np.ndarray:
    """Extract a concatenated feature vector for a protein target.

    Args:
        sequence: Protein amino acid sequence (single-letter codes).
        max_seq_length: Maximum sequence length for normalisation.
        use_amino_acid_composition: Whether to include amino acid composition.

    Returns:
        1-D numpy float32 feature array.
    """
    features: List[np.ndarray] = []

    if use_amino_acid_composition:
        features.append(compute_amino_acid_composition(sequence))

    features.append(compute_physicochemical_features(sequence))
    features.append(compute_sequence_length_features(sequence, max_seq_length))

    return np.concatenate(features, axis=0)


def get_target_feature_dim(use_amino_acid_composition: bool = True) -> int:
    """Return the dimensionality of the target feature vector.

    Args:
        use_amino_acid_composition: Whether amino acid composition is included.

    Returns:
        Total feature dimension as an integer.
    """
    dim = 4 + 2  # physicochemical + length features
    if use_amino_acid_composition:
        dim += 20
    return dim
