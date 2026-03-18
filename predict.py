"""Entry point for inference with a trained DGE-DTI model."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from features.drug_features import extract_drug_features, get_drug_feature_dim
from features.target_features import extract_target_features, get_target_feature_dim
from models.dge_dti import DGEDTI
from utils.helpers import get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict drug-target interaction probabilities.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--input", type=str, required=True, help="CSV file with drug_smiles and target_sequence columns.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file path.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging()
    set_seed(config["training"]["seed"])

    device_str = args.device or get_device()
    device = torch.device(device_str)

    drug_cfg = config["drug_features"]
    target_cfg = config["target_features"]
    model_cfg = config["model"]

    drug_dim = get_drug_feature_dim(drug_cfg["morgan_bits"], drug_cfg["use_maccs"])
    target_dim = get_target_feature_dim(target_cfg["use_amino_acid_composition"])

    model = DGEDTI(
        drug_input_dim=drug_dim,
        target_input_dim=target_dim,
        drug_hidden_dim=model_cfg["drug_hidden_dim"],
        target_hidden_dim=model_cfg["target_hidden_dim"],
        embedding_dim=model_cfg["output_dim"],
        interaction_hidden_dim=model_cfg["interaction_hidden_dim"],
        dropout=0.0,  # No dropout during inference
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    df = pd.read_csv(args.input)
    smiles_col = config["data"]["drug_smiles_col"]
    seq_col = config["data"]["target_sequence_col"]

    predictions = []
    with torch.no_grad():
        for _, row in df.iterrows():
            d_feat = extract_drug_features(
                str(row[smiles_col]),
                morgan_radius=drug_cfg["morgan_radius"],
                morgan_bits=drug_cfg["morgan_bits"],
                use_maccs=drug_cfg["use_maccs"],
            )
            t_feat = extract_target_features(
                str(row[seq_col]),
                max_seq_length=target_cfg["max_seq_length"],
                use_amino_acid_composition=target_cfg["use_amino_acid_composition"],
            )
            d_tensor = torch.tensor(d_feat, dtype=torch.float32).unsqueeze(0).to(device)
            t_tensor = torch.tensor(t_feat, dtype=torch.float32).unsqueeze(0).to(device)
            prob = model(d_tensor, t_tensor).item()
            predictions.append(prob)

    df["predicted_interaction_prob"] = predictions
    df["predicted_label"] = (np.array(predictions) >= 0.5).astype(int)
    df.to_csv(args.output, index=False)
    logger.info("Saved predictions to %s", args.output)


if __name__ == "__main__":
    main()
