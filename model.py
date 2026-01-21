import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512,256), out_dim=256):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GAPDGE(nn.Module):
    def __init__(
        self,
        cell_in,
        drug_in,
        dose_in,
        gene_in,
        latent_dim=256
    ):
        super().__init__()


        self.cell_encoder = MLPEncoder(cell_in, out_dim=latent_dim)
        self.drug_encoder = MLPEncoder(drug_in, out_dim=latent_dim)
        self.dose_encoder = MLPEncoder(dose_in, out_dim=latent_dim)
        self.gene_encoder = MLPEncoder(gene_in, out_dim=latent_dim)


        self.proj = nn.Linear(latent_dim, latent_dim)


        self.q_drug = nn.Linear(latent_dim, latent_dim)
        self.q_cell = nn.Linear(latent_dim, latent_dim)
        self.k_gene = nn.Linear(latent_dim, latent_dim)
        self.v_gene = nn.Linear(latent_dim, latent_dim)


        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 3, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )


    def gene_attention(self, query, gene_emb):

        Q = self.q_drug(query).unsqueeze(1)     
        K = self.k_gene(gene_emb).unsqueeze(0) 
        V = self.v_gene(gene_emb).unsqueeze(0)  

        attn = torch.softmax(
            torch.sum(Q * K, dim=-1) / (query.shape[-1] ** 0.5),
            dim=-1
        )  
        out = torch.sum(attn.unsqueeze(-1) * V, dim=1)  # (B, d)
        return out


    def forward(self, x_ctrl, drug_feat, dose_feat, gene_feat):

        z_cell = self.proj(self.cell_encoder(x_ctrl))       
        z_drug = self.drug_encoder(drug_feat)               
        z_dose = torch.sigmoid(self.dose_encoder(dose_feat))
        z_drug = self.proj(z_drug * z_dose)                 
        gene_feat = gene_feat[0]
        z_gene = self.proj(self.gene_encoder(gene_feat))  

        z_drug_ctx = self.gene_attention(z_drug, z_gene)
        z_cell_ctx = self.gene_attention(z_cell, z_gene)

        B, G = x_ctrl.shape
        z_drug_expand = z_drug_ctx.unsqueeze(1).repeat(1, G, 1)
        z_cell_expand = z_cell_ctx.unsqueeze(1).repeat(1, G, 1)
        z_gene_expand = z_gene.unsqueeze(0).repeat(B, 1, 1)

        h = torch.cat(
            [z_drug_expand, z_gene_expand, z_cell_expand],
            dim=-1
        )

        y_pred = self.predictor(h).squeeze(-1)

        return {
            "y_pred": y_pred,
            "zc_ctrl": z_cell_ctx,
            "zd": z_drug_ctx,
        }
