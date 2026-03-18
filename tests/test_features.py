"""Tests for drug and target feature extraction."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.drug_features import (
    compute_morgan_fingerprint,
    extract_drug_features,
    get_drug_feature_dim,
)
from features.target_features import (
    compute_amino_acid_composition,
    compute_physicochemical_features,
    compute_sequence_length_features,
    extract_target_features,
    get_target_feature_dim,
)

# Aspirin SMILES
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
PROTEIN_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD"


class TestDrugFeatures:
    def test_morgan_fingerprint_shape(self):
        fp = compute_morgan_fingerprint(ASPIRIN_SMILES, radius=2, n_bits=2048)
        assert fp is not None
        assert fp.shape == (2048,)
        assert fp.dtype == np.float32

    def test_morgan_fingerprint_invalid_smiles(self):
        fp = compute_morgan_fingerprint("not_a_smiles")
        assert fp is None

    def test_extract_drug_features_with_maccs(self):
        feat = extract_drug_features(ASPIRIN_SMILES, morgan_bits=2048, use_maccs=True)
        expected_dim = get_drug_feature_dim(morgan_bits=2048, use_maccs=True)
        assert feat.shape == (expected_dim,)
        assert feat.dtype == np.float32

    def test_extract_drug_features_without_maccs(self):
        feat = extract_drug_features(ASPIRIN_SMILES, morgan_bits=2048, use_maccs=False)
        expected_dim = get_drug_feature_dim(morgan_bits=2048, use_maccs=False)
        assert feat.shape == (expected_dim,)

    def test_drug_feature_dim_with_maccs(self):
        dim = get_drug_feature_dim(morgan_bits=1024, use_maccs=True)
        assert dim == 1024 + 167

    def test_drug_feature_dim_without_maccs(self):
        dim = get_drug_feature_dim(morgan_bits=1024, use_maccs=False)
        assert dim == 1024


class TestTargetFeatures:
    def test_amino_acid_composition_shape(self):
        aac = compute_amino_acid_composition(PROTEIN_SEQ)
        assert aac.shape == (20,)
        assert aac.dtype == np.float32

    def test_amino_acid_composition_sums_to_one(self):
        aac = compute_amino_acid_composition(PROTEIN_SEQ)
        assert abs(aac.sum() - 1.0) < 1e-5

    def test_amino_acid_composition_empty_sequence(self):
        aac = compute_amino_acid_composition("")
        assert aac.shape == (20,)
        assert aac.sum() == 0.0

    def test_physicochemical_features_shape(self):
        phys = compute_physicochemical_features(PROTEIN_SEQ)
        assert phys.shape == (4,)
        assert phys.dtype == np.float32

    def test_sequence_length_features(self):
        feat = compute_sequence_length_features(PROTEIN_SEQ, max_length=1000)
        assert feat.shape == (2,)
        assert 0.0 <= feat[0] <= 1.0

    def test_extract_target_features_with_aac(self):
        feat = extract_target_features(PROTEIN_SEQ, use_amino_acid_composition=True)
        expected_dim = get_target_feature_dim(use_amino_acid_composition=True)
        assert feat.shape == (expected_dim,)
        assert feat.dtype == np.float32

    def test_extract_target_features_without_aac(self):
        feat = extract_target_features(PROTEIN_SEQ, use_amino_acid_composition=False)
        expected_dim = get_target_feature_dim(use_amino_acid_composition=False)
        assert feat.shape == (expected_dim,)

    def test_target_feature_dim_with_aac(self):
        dim = get_target_feature_dim(use_amino_acid_composition=True)
        assert dim == 20 + 4 + 2

    def test_target_feature_dim_without_aac(self):
        dim = get_target_feature_dim(use_amino_acid_composition=False)
        assert dim == 4 + 2
