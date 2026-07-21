# GAPDGE

Gene-Aware Drug Perturbed Gene Expression Prediction.

GAPDGE is designed to predict drug-perturbed gene expression by integrating cellular state, drug representation, dosage information, and gene embeddings. The model uses a gene-aware attention mechanism to capture gene-specific response patterns and a contrastive alignment objective to improve the consistency between cellular and drug representations.

## Project Structure

```text
GAPDGE-NEW/
├── data/
│   └── README.md              # Data placement instructions
├── output/
│   └── README.md              # Output directory instructions
├── src/gapdge/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── dataset.py             # Dataset construction and data splits
│   ├── loss.py                # Prediction loss and contrastive loss
│   ├── model.py               # GAPDGE model
│   ├── train.py               # Training and evaluation loop
│   └── utils.py               # Checkpoint utilities
├── main.py                    # Main entry point
├── README.md
└── requirements.txt
```

## Environment

Create and activate a Python environment:

```bash
conda create -n GAPDGE python=3.9
conda activate GAPDGE
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Download the required data and pretrained embeddings from:

```text
https://cloud.tsinghua.edu.cn/f/7bca2e22c1f14c4db7db/?dl=1
```

Place the following files in `data/`:

```text
data/
├── lincs_adata.h5ad
├── pert_smiles_emb.pkl
├── dosage_prompt_emb_lincs.pkl
└── gene_embeddings.npy
```

These data files are not included in this repository because they are large.

## Train and Evaluate

Run with the default paths:

```bash
python main.py
```

The default command expects the required files to be placed in `data/`.

You can also specify paths and training options manually:

```bash
python main.py \
  --adata data/lincs_adata.h5ad \
  --dose_emb data/dosage_prompt_emb_lincs.pkl \
  --drug_emb data/pert_smiles_emb.pkl \
  --gene_emb data/gene_embeddings.npy \
  --split_key Drug_unseen \
  --epochs 100 \
  --bs 64 \
  --device cuda:0
```

Training logs and model checkpoints are saved to:

```text
output/<split_key>/
├── best_model_<timestamp>.pth
└── train_<timestamp>.log
```

## Arguments

Commonly used arguments:

```text
--adata          Path to the AnnData file.
--dose_emb       Path to dosage prompt embeddings.
--drug_emb       Path to drug/SMILES embeddings.
--gene_emb       Path to gene embeddings.
--out            Output directory.
--bs             Batch size.
--epochs         Number of training epochs.
--latent         Latent dimension.
--split_key      Split column in adata.obs.
--lr             Learning rate.
--lambda_cl      Weight of the contrastive loss.
--temp           Temperature for contrastive learning.
--device         Device, such as cuda:0 or cpu.
--num_workers    Number of DataLoader workers.
```

## Split Keys

The selected split key must exist in `adata.obs`. Common options include:

```text
Both_unseen
Cell_line_unseen
Drug_unseen
```
