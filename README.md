# DGE-DTI: Drug Graph Encoder for Drug-Target Interaction Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

DGE-DTI is a deep learning framework for predicting drug-target interactions (DTIs). It uses a **Drug Graph Encoder** that encodes molecular fingerprints and a **Target Encoder** that encodes protein sequence features, then combines them through an interaction predictor to estimate the probability of a drug binding to a protein target.

---

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Benchmark Datasets](#benchmark-datasets)
- [Contributing](#contributing)

---

## Background

Drug-target interaction (DTI) prediction is a fundamental problem in computational drug discovery. Identifying which drugs bind to which protein targets can:

- Accelerate drug repurposing for new indications
- Reduce experimental costs in the early stages of drug development
- Help predict off-target effects and drug toxicity
- Guide lead optimisation in medicinal chemistry

DGE-DTI approaches this problem using molecular fingerprints (Morgan/circular fingerprints and MACCS keys) to represent drugs and sequence-derived features (amino acid composition and physicochemical properties) to represent protein targets. Both representations are mapped into a shared latent space, where a feed-forward network predicts interaction probability.

---

## Project Structure

```
DGE-DTI/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── train.py                    # Training entry point
├── predict.py                  # Inference entry point
├── configs/
│   └── config.yaml             # Hyperparameters and data paths
├── data/
│   ├── README.md               # Data documentation
│   └── sample_data.csv         # Sample dataset
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   └── preprocessing.py    # Data validation and splitting
│   ├── features/
│   │   ├── drug_features.py    # Molecular fingerprint extraction
│   │   └── target_features.py  # Protein feature extraction
│   ├── models/
│   │   ├── dge_dti.py          # DGE-DTI model (main architecture)
│   │   └── baseline.py         # Baseline MLP model
│   ├── training/
│   │   ├── trainer.py          # Training loop with early stopping
│   │   └── metrics.py          # Evaluation metrics (AUROC, AUPRC, F1, ...)
│   └── utils/
│       └── helpers.py          # Seeding, config loading, device utilities
└── tests/
    ├── test_features.py        # Feature extraction tests
    ├── test_models.py          # Model architecture tests
    ├── test_dataset.py         # Data loading tests
    └── test_metrics.py         # Metrics tests
```

---

## Architecture

```
Drug SMILES  ─────►  Morgan FP + MACCS  ──►  DrugEncoder  ──►  drug_emb ─────┐
                                                                                ├──► InteractionPredictor ──► P(interaction)
Protein Seq  ─────►  AAC + Physicochemical  ──►  TargetEncoder  ──►  target_emb ─┘
```

### Drug Encoder
- Encodes **Morgan circular fingerprints** (2048 bits, radius 2) and optional **MACCS keys** (167 bits)
- Three-layer MLP with BatchNorm, ReLU, and Dropout

### Target Encoder  
- Encodes **amino acid composition** (20-dimensional), **mean physicochemical properties** (hydrophobicity, charge, polarity, MW), and **sequence length features**
- Three-layer MLP with BatchNorm, ReLU, and Dropout

### Interaction Predictor
- Concatenates drug and target embeddings
- Three-layer MLP → Sigmoid to produce interaction probability in [0, 1]

---

## Installation

### Prerequisites

- Python 3.8 or newer
- (Optional) CUDA-capable GPU for accelerated training

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/mingxinDonald/DGE-DTI.git
cd DGE-DTI

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode (optional)
pip install -e .
```

---

## Quick Start

```python
import sys
sys.path.insert(0, "src")

from features.drug_features import extract_drug_features
from features.target_features import extract_target_features
from models.dge_dti import DGEDTI
import torch

# --- Define drug and target ---
smiles   = "CC(=O)Oc1ccccc1C(=O)O"          # Aspirin
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGL"   # Partial protein sequence

# --- Extract features ---
drug_feat   = extract_drug_features(smiles)          # shape: (2215,)
target_feat = extract_target_features(sequence)      # shape: (26,)

# --- Build model ---
model = DGEDTI(
    drug_input_dim=len(drug_feat),
    target_input_dim=len(target_feat),
)
model.eval()

# --- Predict interaction probability ---
d = torch.tensor(drug_feat).unsqueeze(0)
t = torch.tensor(target_feat).unsqueeze(0)
with torch.no_grad():
    prob = model(d, t).item()

print(f"Interaction probability: {prob:.4f}")
```

---

## Training

### 1. Prepare your data

Place your dataset CSV in `data/` and update `configs/config.yaml`:

```yaml
data:
  data_path: "data/your_dataset.csv"
  drug_smiles_col: "drug_smiles"
  target_sequence_col: "target_sequence"
  label_col: "label"
```

The CSV must contain at minimum:

| Column | Description |
|--------|-------------|
| `drug_smiles` | SMILES string of the drug |
| `target_sequence` | Protein amino acid sequence |
| `label` | Binary label: `1` (interaction) or `0` (no interaction) |

### 2. Run training

```bash
# Train the DGE-DTI model (default)
python train.py --config configs/config.yaml --model dge_dti

# Train the baseline MLP
python train.py --config configs/config.yaml --model baseline

# Force CPU training
python train.py --device cpu
```

Model checkpoints and results are saved to `outputs/` by default.

---

## Inference

```bash
python predict.py \
    --config configs/config.yaml \
    --checkpoint outputs/models/best_model.pt \
    --input data/my_pairs.csv \
    --output predictions.csv
```

The output CSV will contain the original columns plus:
- `predicted_interaction_prob`: Probability score in [0, 1]
- `predicted_label`: Binary prediction at threshold 0.5

---

## Data Format

### Input CSV

```csv
drug_smiles,target_sequence,label,drug_name,target_name
CC(=O)Oc1ccccc1C(=O)O,MKTAYIAKQRQISFVK...,1,Aspirin,COX-2
CC(C)Cc1ccc(cc1)C(C)C(=O)O,MSALGAVIALLLW...,1,Ibuprofen,Albumin
```

### Feature Dimensions (defaults)

| Feature | Dimension |
|---------|-----------|
| Morgan fingerprint (r=2, 2048 bits) | 2048 |
| MACCS keys | 167 |
| **Total drug features** | **2215** |
| Amino acid composition | 20 |
| Physicochemical properties | 4 |
| Sequence length features | 2 |
| **Total target features** | **26** |

---

## Configuration

All hyperparameters are controlled via `configs/config.yaml`:

```yaml
model:
  drug_hidden_dim: 256        # Drug encoder hidden layer size
  target_hidden_dim: 256      # Target encoder hidden layer size
  interaction_hidden_dim: 512 # Interaction MLP hidden layer size
  output_dim: 128             # Embedding dimensionality
  dropout: 0.2                # Dropout probability

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 15
  val_split: 0.1
  test_split: 0.1
  seed: 42
```

---

## Evaluation Metrics

The framework reports the following metrics on validation and test sets:

| Metric | Description |
|--------|-------------|
| **AUROC** | Area Under the ROC Curve — primary ranking metric |
| **AUPRC** | Area Under the Precision-Recall Curve — important for imbalanced data |
| **Accuracy** | Fraction of correct predictions at threshold 0.5 |
| **Precision** | True positive rate among predicted positives |
| **Recall** | True positive rate among actual positives |
| **F1 Score** | Harmonic mean of precision and recall |

---

## Benchmark Datasets

For rigorous evaluation consider the following public DTI benchmarks:

| Dataset | Description | Link |
|---------|-------------|------|
| **BindingDB** | Measured binding affinities | https://www.bindingdb.org/ |
| **ChEMBL** | Bioactivity data for drug-like molecules | https://www.ebi.ac.uk/chembl/ |
| **DrugBank** | Approved and experimental drugs | https://go.drugbank.com/ |
| **DAVIS** | Kinase inhibitor selectivity (Kd values) | — |
| **KIBA** | Kinase inhibitor bioactivity scores | — |
| **DUD-E** | Directory of Useful Decoys Enhanced | https://dude.docking.org/ |

---

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_models.py -v
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Submit a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
