# Data Directory

This directory contains datasets for Drug-Target Interaction (DTI) prediction.

## Sample Data

`sample_data.csv` contains a small example dataset with the following columns:

| Column | Description |
|--------|-------------|
| `drug_smiles` | SMILES string representation of the drug molecule |
| `target_sequence` | Amino acid sequence of the protein target |
| `label` | Binary interaction label (1 = interacts, 0 = no interaction) |
| `drug_name` | Human-readable drug name |
| `target_name` | Human-readable target name |

## Benchmark Datasets

For real experiments, consider using the following publicly available DTI datasets:

- **BindingDB**: https://www.bindingdb.org/
- **ChEMBL**: https://www.ebi.ac.uk/chembl/
- **DrugBank**: https://go.drugbank.com/
- **DAVIS**: Kinase inhibitor bioactivity dataset
- **KIBA**: Kinase Inhibitor BioActivity dataset

## Data Format

Place your dataset CSV files in this directory and update `configs/config.yaml` 
with the correct `data_path` and column names.
