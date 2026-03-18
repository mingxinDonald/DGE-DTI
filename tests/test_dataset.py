"""Tests for data loading and preprocessing utilities."""

import sys
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.preprocessing import (
    filter_dataset,
    normalise_labels,
    split_indices,
    validate_sequence,
    validate_smiles,
)
from data.dataset import DTIDataset

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
PROTEIN_SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD"
PROTEIN_SEQ_B = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small sample CSV for testing."""
    df = pd.DataFrame(
        {
            "drug_smiles": [ASPIRIN_SMILES, IBUPROFEN_SMILES],
            "target_sequence": [PROTEIN_SEQ_A, PROTEIN_SEQ_B],
            "label": [1, 0],
        }
    )
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestValidation:
    def test_valid_smiles(self):
        assert validate_smiles(ASPIRIN_SMILES) is True

    def test_invalid_smiles(self):
        assert validate_smiles("!!invalid!!") is False

    def test_valid_sequence(self):
        assert validate_sequence(PROTEIN_SEQ_A) is True

    def test_sequence_too_short(self):
        assert validate_sequence("ACK") is False

    def test_sequence_with_invalid_chars(self):
        # More than 10% invalid characters
        assert validate_sequence("BBBBBBBBBBB") is False


class TestFilterDataset:
    def test_filter_keeps_valid_rows(self):
        df = pd.DataFrame(
            {
                "drug_smiles": [ASPIRIN_SMILES, "invalid_smiles"],
                "target_sequence": [PROTEIN_SEQ_A, PROTEIN_SEQ_B],
                "label": [1, 0],
            }
        )
        filtered = filter_dataset(df)
        assert len(filtered) == 1
        assert filtered.iloc[0]["label"] == 1


class TestSplitIndices:
    def test_split_counts(self):
        train, val, test = split_indices(100, val_split=0.1, test_split=0.1, seed=0)
        assert len(test) == 10
        total = len(train) + len(val) + len(test)
        assert total == 100

    def test_no_overlap(self):
        train, val, test = split_indices(100, val_split=0.1, test_split=0.1, seed=0)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0


class TestNormaliseLabels:
    def test_range_normalisation(self):
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalised, min_v, max_v = normalise_labels(labels)
        assert abs(normalised.min() - 0.0) < 1e-6
        assert abs(normalised.max() - 1.0) < 1e-6
        assert min_v == 1.0
        assert max_v == 5.0

    def test_constant_labels(self):
        labels = np.array([3.0, 3.0, 3.0])
        normalised, _, _ = normalise_labels(labels)
        assert (normalised == 0).all()


class TestDTIDataset:
    def test_dataset_length(self, sample_csv):
        ds = DTIDataset(sample_csv, morgan_bits=256, use_maccs=False)
        assert len(ds) == 2

    def test_dataset_item_keys(self, sample_csv):
        ds = DTIDataset(sample_csv, morgan_bits=256, use_maccs=False)
        item = ds[0]
        assert "drug_features" in item
        assert "target_features" in item
        assert "label" in item

    def test_dataset_feature_shapes(self, sample_csv):
        ds = DTIDataset(sample_csv, morgan_bits=256, use_maccs=False, use_amino_acid_composition=True)
        item = ds[0]
        assert item["drug_features"].shape[0] == ds.drug_feature_dim
        assert item["target_features"].shape[0] == ds.target_feature_dim

    def test_dataset_labels(self, sample_csv):
        ds = DTIDataset(sample_csv, morgan_bits=256, use_maccs=False)
        labels = ds.get_labels()
        assert set(labels) <= {0.0, 1.0}
