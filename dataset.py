import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import scanpy as sc
from scipy import sparse

def train_valid_test(adata, split_key = 'cell_type_split_0'):
    train_index = adata.obs[(adata.obs[split_key] == 'train') & (adata.obs['dose']!= 0.0)].index.tolist()
    valid_index = adata.obs[(adata.obs[split_key] == 'valid') & (adata.obs['dose']!= 0.0)].index.tolist()
    test_index = adata.obs[(adata.obs[split_key] == 'test') & (adata.obs['dose']!= 0.0)].index.tolist()
    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()

    if len(train_index)>0:
        train_index = train_index + control_index
        train_adata = adata[train_index, :].copy()
    else:
        train_adata = None
    if len(valid_index)>0:
        valid_index = valid_index + control_index
        valid_adata = adata[valid_index, :].copy()
    else:
        valid_adata=None
    if len(test_index)>0:
        test_index = test_index + control_index
        test_adata = adata[test_index, :].copy()
    else:
        test_adata=None
    print(f'The size of train_data: {len(train_index)}; valid_data: {len(valid_index)}; test_data: {len(test_index)}; control_data: {len(control_index)}')
    return train_adata, valid_adata, test_adata


class PerturbationDataset(Dataset):
    def __init__(self, adata, gene_feat_matrix=None, drug_feat_lookup=None, dosage_prompt_emb=None, condition_key='condition', control_dose=0.0):
        self.obs = adata.obs.reset_index(drop=True)
        self.gene_feat_matrix = gene_feat_matrix
        self.drug_feat_lookup = drug_feat_lookup
        self.condition_key = condition_key
        self.control_dose = control_dose
        self.dense_adata = adata
        self.drug_adata = self.dense_adata[self.dense_adata.obs['dose']!=self.control_dose] 
        self.data = torch.tensor(self.drug_adata.X, dtype=torch.float32)
        self.dense_data = torch.tensor(self.dense_adata.X, dtype=torch.float32)
        self.paired_control_index = self.drug_adata.obs['paired_control_index'].tolist()
        self.dense_adata_index = self.dense_adata.obs.index.to_list()
        self.control_index_dict = {index: i for i, index in enumerate(self.dense_adata_index)}
        self.drug_type_list = self.drug_adata.obs['SMILES'].to_list()
        self.dose_list = self.drug_adata.obs['dose_val_4f'].to_list()
        self.dosage_prompt_emb = dosage_prompt_emb

    def __len__(self):
        return len(self.drug_adata)

    def __getitem__(self, idx):

        y = self.data[idx, :]
        control_index = self.control_index_dict[self.paired_control_index[idx]]   
        x = self.dense_data[control_index, :]
        drug_feat = self.drug_feat_lookup[self.drug_type_list[idx]].mean(dim=0)
        dose_feat = self.dosage_prompt_emb[self.dose_list[idx]].mean(dim=0)
        gene_feat = self.gene_feat_matrix

        return y, x, dose_feat, drug_feat, torch.from_numpy(gene_feat)#.mean(dim=1)
