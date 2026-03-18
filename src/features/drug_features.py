"""Drug feature extraction from SMILES strings."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Compute Morgan (circular) fingerprint for a drug molecule.

    Args:
        smiles: SMILES string of the drug molecule.
        radius: Radius of the Morgan fingerprint.
        n_bits: Number of bits in the fingerprint.

    Returns:
        Numpy array of shape (n_bits,) or None if parsing fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Could not parse SMILES: %s", smiles)
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except ImportError:
        logger.warning("RDKit not available; falling back to zero fingerprint.")
        return np.zeros(n_bits, dtype=np.float32)


def compute_maccs_keys(smiles: str) -> Optional[np.ndarray]:
    """Compute MACCS keys fingerprint (166 bits) for a drug molecule.

    Args:
        smiles: SMILES string of the drug molecule.

    Returns:
        Numpy array of shape (167,) or None if parsing fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Could not parse SMILES: %s", smiles)
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.float32)
    except ImportError:
        logger.warning("RDKit not available; falling back to zero MACCS keys.")
        return np.zeros(167, dtype=np.float32)


def extract_drug_features(
    smiles: str,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    use_maccs: bool = True,
) -> np.ndarray:
    """Extract a concatenated feature vector for a drug.

    Args:
        smiles: SMILES string of the drug molecule.
        morgan_radius: Radius for Morgan fingerprint.
        morgan_bits: Number of bits for Morgan fingerprint.
        use_maccs: Whether to include MACCS keys in the feature vector.

    Returns:
        1-D numpy float32 array of drug features.
    """
    features: List[np.ndarray] = []

    morgan = compute_morgan_fingerprint(smiles, radius=morgan_radius, n_bits=morgan_bits)
    if morgan is None:
        morgan = np.zeros(morgan_bits, dtype=np.float32)
    features.append(morgan)

    if use_maccs:
        maccs = compute_maccs_keys(smiles)
        if maccs is None:
            maccs = np.zeros(167, dtype=np.float32)
        features.append(maccs)

    return np.concatenate(features, axis=0)


def get_drug_feature_dim(morgan_bits: int = 2048, use_maccs: bool = True) -> int:
    """Return the dimensionality of the drug feature vector.

    Args:
        morgan_bits: Number of bits for Morgan fingerprint.
        use_maccs: Whether MACCS keys are included.

    Returns:
        Total feature dimension as an integer.
    """
    dim = morgan_bits
    if use_maccs:
        dim += 167
    return dim
