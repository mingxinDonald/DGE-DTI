"""Tests for DGE-DTI model architectures."""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.baseline import BaselineMLP
from models.dge_dti import DGEDTI, DrugEncoder, TargetEncoder, InteractionPredictor

DRUG_DIM = 64
TARGET_DIM = 26
BATCH_SIZE = 4


def make_drug_input(batch=BATCH_SIZE):
    return torch.randn(batch, DRUG_DIM)


def make_target_input(batch=BATCH_SIZE):
    return torch.randn(batch, TARGET_DIM)


class TestDrugEncoder:
    def test_output_shape(self):
        enc = DrugEncoder(input_dim=DRUG_DIM, hidden_dim=32, output_dim=16)
        out = enc(make_drug_input())
        assert out.shape == (BATCH_SIZE, 16)

    def test_single_sample(self):
        enc = DrugEncoder(input_dim=DRUG_DIM, hidden_dim=32, output_dim=16)
        enc.eval()
        out = enc(make_drug_input(batch=1))
        assert out.shape == (1, 16)

    def test_output_non_negative(self):
        enc = DrugEncoder(input_dim=DRUG_DIM, hidden_dim=32, output_dim=16)
        out = enc(make_drug_input())
        assert (out >= 0).all(), "ReLU output should be non-negative"


class TestTargetEncoder:
    def test_output_shape(self):
        enc = TargetEncoder(input_dim=TARGET_DIM, hidden_dim=32, output_dim=16)
        out = enc(make_target_input())
        assert out.shape == (BATCH_SIZE, 16)


class TestInteractionPredictor:
    def test_output_shape(self):
        pred = InteractionPredictor(drug_dim=16, target_dim=16, hidden_dim=32)
        drug_emb = torch.randn(BATCH_SIZE, 16)
        target_emb = torch.randn(BATCH_SIZE, 16)
        out = pred(drug_emb, target_emb)
        assert out.shape == (BATCH_SIZE, 1)

    def test_output_in_range(self):
        pred = InteractionPredictor(drug_dim=16, target_dim=16, hidden_dim=32)
        drug_emb = torch.randn(BATCH_SIZE, 16)
        target_emb = torch.randn(BATCH_SIZE, 16)
        out = pred(drug_emb, target_emb)
        assert (out >= 0).all() and (out <= 1).all(), "Sigmoid output must be in [0, 1]"


class TestDGEDTI:
    def test_output_shape(self):
        model = DGEDTI(
            drug_input_dim=DRUG_DIM,
            target_input_dim=TARGET_DIM,
            drug_hidden_dim=32,
            target_hidden_dim=32,
            embedding_dim=16,
            interaction_hidden_dim=32,
        )
        out = model(make_drug_input(), make_target_input())
        assert out.shape == (BATCH_SIZE, 1)

    def test_output_probabilities(self):
        model = DGEDTI(
            drug_input_dim=DRUG_DIM,
            target_input_dim=TARGET_DIM,
            drug_hidden_dim=32,
            target_hidden_dim=32,
            embedding_dim=16,
            interaction_hidden_dim=32,
        )
        out = model(make_drug_input(), make_target_input())
        assert (out >= 0).all() and (out <= 1).all()

    def test_encode_drug(self):
        model = DGEDTI(
            drug_input_dim=DRUG_DIM,
            target_input_dim=TARGET_DIM,
            drug_hidden_dim=32,
            target_hidden_dim=32,
            embedding_dim=16,
            interaction_hidden_dim=32,
        )
        emb = model.encode_drug(make_drug_input())
        assert emb.shape == (BATCH_SIZE, 16)

    def test_encode_target(self):
        model = DGEDTI(
            drug_input_dim=DRUG_DIM,
            target_input_dim=TARGET_DIM,
            drug_hidden_dim=32,
            target_hidden_dim=32,
            embedding_dim=16,
            interaction_hidden_dim=32,
        )
        emb = model.encode_target(make_target_input())
        assert emb.shape == (BATCH_SIZE, 16)


class TestBaselineMLP:
    def test_output_shape(self):
        model = BaselineMLP(
            drug_input_dim=DRUG_DIM,
            target_input_dim=TARGET_DIM,
            hidden_dim=32,
        )
        out = model(make_drug_input(), make_target_input())
        assert out.shape == (BATCH_SIZE, 1)

    def test_output_probabilities(self):
        model = BaselineMLP(
            drug_input_dim=DRUG_DIM,
            target_input_dim=TARGET_DIM,
            hidden_dim=32,
        )
        out = model(make_drug_input(), make_target_input())
        assert (out >= 0).all() and (out <= 1).all()
