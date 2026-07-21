import argparse
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def build_dataloader(dataset, batch_size, shuffle, num_workers):
    import torch
    from torch.utils.data import DataLoader

    if dataset is None:
        return None

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(dataset, **loader_kwargs)


def resolve_device(device_arg):
    import torch

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    h5ad_path,
    dose_emb,
    drug_emb,
    gene_emb,
    out="./output",
    bs=64,
    epochs=30,
    latent=256,
    split_key="Both_unseen",
    lr=1e-4,
    lambda_cl=0.01,
    temp=0.1,
    device=None,
    num_workers=8,
):
    import numpy as np
    import scanpy as sc
    import torch

    from gapdge.dataset import PerturbationDataset, train_valid_test
    from gapdge.model import GAPDGE
    from gapdge.train import Trainer

    print("Loading data")
    drug_lookup = torch.load(drug_emb, weights_only=False)
    dosage_prompt_emb = torch.load(dose_emb, weights_only=False)
    gene_emb_matrix = np.load(gene_emb)

    adata = sc.read_h5ad(h5ad_path)
    adata.X = np.clip(adata.X, 0, 1e3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    train_adata, valid_adata, test_adata = train_valid_test(adata, split_key=split_key)
    train_ds = PerturbationDataset(
        train_adata,
        gene_feat_matrix=gene_emb_matrix,
        drug_feat_lookup=drug_lookup,
        dosage_prompt_emb=dosage_prompt_emb,
        condition_key="condition",
    )
    valid_ds = (
        PerturbationDataset(
            valid_adata,
            gene_feat_matrix=gene_emb_matrix,
            drug_feat_lookup=drug_lookup,
            dosage_prompt_emb=dosage_prompt_emb,
            condition_key="condition",
        )
        if valid_adata is not None
        else None
    )
    test_ds = (
        PerturbationDataset(
            test_adata,
            gene_feat_matrix=gene_emb_matrix,
            drug_feat_lookup=drug_lookup,
            dosage_prompt_emb=dosage_prompt_emb,
            condition_key="condition",
        )
        if test_adata is not None
        else None
    )

    train_loader = build_dataloader(train_ds, bs, shuffle=True, num_workers=num_workers)
    valid_loader = build_dataloader(valid_ds, bs, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_ds, bs, shuffle=False, num_workers=num_workers)

    y, x, dose, drug, gene = train_ds[0]
    model = GAPDGE(
        cell_in=y.shape[0],
        dose_in=dose.shape[0],
        drug_in=drug.shape[0],
        gene_in=gene.shape[1],
        latent_dim=latent,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=resolve_device(device),
        out_dir=out,
        split_key=split_key,
        cfg={"epochs": epochs, "lr": lr, "lambda": lambda_cl, "temp": temp},
    )
    trainer.fit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate GAPDGE for drug-perturbed gene expression prediction."
    )
    parser.add_argument("--adata", type=str, default=str(DEFAULT_DATA_DIR / "lincs_adata.h5ad"))
    parser.add_argument("--dose_emb", type=str, default=str(DEFAULT_DATA_DIR / "dosage_prompt_emb_lincs.pkl"))
    parser.add_argument("--drug_emb", type=str, default=str(DEFAULT_DATA_DIR / "pert_smiles_emb.pkl"))
    parser.add_argument("--gene_emb", type=str, default=str(DEFAULT_DATA_DIR / "gene_embeddings.npy"))
    parser.add_argument("--out", type=str, default="./output")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--split_key", type=str, default="Both_unseen")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_cl", type=float, default=0.01)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None, help="Example: cuda, cuda:0, cuda:1, or cpu.")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        h5ad_path=args.adata,
        dose_emb=args.dose_emb,
        drug_emb=args.drug_emb,
        gene_emb=args.gene_emb,
        out=args.out,
        bs=args.bs,
        epochs=args.epochs,
        latent=args.latent,
        split_key=args.split_key,
        lr=args.lr,
        lambda_cl=args.lambda_cl,
        temp=args.temp,
        device=args.device,
        num_workers=args.num_workers,
    )
