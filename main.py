import argparse, os, numpy as np, scanpy as sc, torch
from turtle import shape
from torch.utils.data import DataLoader
from model import GAPDGE
from dataset import PerturbationDataset, train_valid_test
from train import Trainer
import scipy.sparse as sp

def run(h5ad_path, dose_emb=None, drug_emb=None, gene_emb=None, out='./output', bs=64, epochs=30, latent=256, split_key='drug_split_0'):
    print("Loading data")
    drug_lookup = torch.load(drug_emb)
    dosage_prompt_emb = torch.load(dose_emb, weights_only=False)
    gene_emb_matrix = np.load(gene_emb)
    adata = sc.read_h5ad(h5ad_path)
    adata.X = np.clip(adata.X, 0, 1e3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    train_adata, valid_adata, test_adata = train_valid_test(adata, split_key=split_key)

    train_ds = PerturbationDataset(train_adata, gene_feat_matrix=gene_emb_matrix, drug_feat_lookup=drug_lookup, dosage_prompt_emb=dosage_prompt_emb, condition_key='condition')
    valid_ds = PerturbationDataset(valid_adata, gene_feat_matrix=gene_emb_matrix, drug_feat_lookup=drug_lookup, dosage_prompt_emb=dosage_prompt_emb, condition_key='condition') if valid_adata is not None else None
    test_ds = PerturbationDataset(test_adata, gene_feat_matrix=gene_emb_matrix, drug_feat_lookup=drug_lookup, dosage_prompt_emb=dosage_prompt_emb, condition_key='condition') if test_adata is not None else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4) if valid_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4) if test_ds is not None else None


    sample = train_ds[0]
    y, x, dose, drug, gene = sample
    cell_dim = y.shape[0]
    dose_dim = dose.shape[0]
    drug_dim = drug.shape[0]
    gene_dim = gene.shape[1]
    model = GAPDGE(cell_in=cell_dim, dose_in=dose_dim, drug_in=drug_dim, gene_in=gene_dim, latent_dim=latent)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, train_loader, valid_loader, test_loader, device, out, split_key, {'epochs':epochs, 'lr':1e-4, 'lambda':0.01, 'temp':0.1})
    trainer.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adata', type=str, default='/data/lincs_adata.h5ad')
    parser.add_argument('--dose_emb', type=str, default='/data/dosage_prompt_emb_lincs.pkl')
    parser.add_argument('--drug_emb', type=str, default='/data/pert_smiles_emb.pkl')
    parser.add_argument('--gene_emb', type=str, default='/data/gene_embeddings.npy')
    parser.add_argument('--out', type=str, default='./output')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--latent', type=int, default=256)
    parser.add_argument('--split_key', type=str, default='Drug_unseen')
    args = parser.parse_args()
    run(args.adata, dose_emb=args.dose_emb, drug_emb=args.drug_emb, gene_emb=args.gene_emb, out=args.out, bs=args.bs, epochs=args.epochs, latent=args.latent, split_key=args.split_key)
