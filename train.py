"""Entry point for training a DGE-DTI model."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

# Allow running as a script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.dataset import DTIDataset
from data.preprocessing import split_indices
from models.baseline import BaselineMLP
from models.dge_dti import DGEDTI
from training.trainer import Trainer
from utils.helpers import get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Drug-Target Interaction model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dge_dti", "baseline"],
        default="dge_dti",
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g., 'cpu', 'cuda'). Defaults to auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    setup_logging(log_dir=config["output"]["log_dir"])
    logger.info("Loaded config from %s", args.config)

    seed = config["training"]["seed"]
    set_seed(seed)

    device = args.device or get_device()
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Build full dataset to determine feature dimensions
    # ------------------------------------------------------------------
    data_cfg = config["data"]
    drug_cfg = config["drug_features"]
    target_cfg = config["target_features"]

    full_dataset = DTIDataset(
        data_path=data_cfg["data_path"],
        drug_smiles_col=data_cfg["drug_smiles_col"],
        target_sequence_col=data_cfg["target_sequence_col"],
        label_col=data_cfg["label_col"],
        morgan_radius=drug_cfg["morgan_radius"],
        morgan_bits=drug_cfg["morgan_bits"],
        use_maccs=drug_cfg["use_maccs"],
        max_seq_length=target_cfg["max_seq_length"],
        use_amino_acid_composition=target_cfg["use_amino_acid_composition"],
    )
    drug_dim = full_dataset.drug_feature_dim
    target_dim = full_dataset.target_feature_dim

    # ------------------------------------------------------------------
    # Split data
    # ------------------------------------------------------------------
    train_cfg = config["training"]
    train_idx, val_idx, test_idx = split_indices(
        n_samples=len(full_dataset),
        val_split=train_cfg["val_split"],
        test_split=train_cfg["test_split"],
        seed=seed,
    )

    def make_subset(indices):
        return DTIDataset(
            data_path=data_cfg["data_path"],
            drug_smiles_col=data_cfg["drug_smiles_col"],
            target_sequence_col=data_cfg["target_sequence_col"],
            label_col=data_cfg["label_col"],
            morgan_radius=drug_cfg["morgan_radius"],
            morgan_bits=drug_cfg["morgan_bits"],
            use_maccs=drug_cfg["use_maccs"],
            max_seq_length=target_cfg["max_seq_length"],
            use_amino_acid_composition=target_cfg["use_amino_acid_composition"],
            indices=indices,
        )

    train_dataset = make_subset(train_idx)
    val_dataset = make_subset(val_idx)
    test_dataset = make_subset(test_idx)

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model_cfg = config["model"]
    if args.model == "dge_dti":
        model = DGEDTI(
            drug_input_dim=drug_dim,
            target_input_dim=target_dim,
            drug_hidden_dim=model_cfg["drug_hidden_dim"],
            target_hidden_dim=model_cfg["target_hidden_dim"],
            embedding_dim=model_cfg["output_dim"],
            interaction_hidden_dim=model_cfg["interaction_hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    else:
        model = BaselineMLP(
            drug_input_dim=drug_dim,
            target_input_dim=target_dim,
            hidden_dim=model_cfg["interaction_hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    logger.info("Model: %s | Parameters: %d", args.model, sum(p.numel() for p in model.parameters()))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        device=device,
        model_dir=config["output"]["model_dir"],
    )
    trainer.train(
        train_loader,
        val_loader,
        epochs=train_cfg["epochs"],
        early_stopping_patience=train_cfg["early_stopping_patience"],
    )

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    trainer.load_checkpoint("best_model.pt")
    test_metrics = trainer.evaluate(test_loader)
    logger.info("Test metrics: %s", test_metrics)

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "test_metrics.txt")
    with open(results_path, "w", encoding="utf-8") as fh:
        for k, v in test_metrics.items():
            fh.write(f"{k}: {v:.4f}\n")
    logger.info("Saved test metrics to %s", results_path)


if __name__ == "__main__":
    main()
