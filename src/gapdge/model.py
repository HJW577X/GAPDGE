import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), out_dim=256):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.GELU()])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GAPDGE(nn.Module):
    def __init__(self, cell_in, drug_in, dose_in, gene_in, latent_dim=256):
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
            nn.Linear(512, 1),
        )

    def gene_attention(self, query, gene_emb):
        query_proj = self.q_drug(query).unsqueeze(1)
        key = self.k_gene(gene_emb).unsqueeze(0)
        value = self.v_gene(gene_emb).unsqueeze(0)

        attention = torch.softmax(
            torch.sum(query_proj * key, dim=-1) / (query.shape[-1] ** 0.5),
            dim=-1,
        )
        return torch.sum(attention.unsqueeze(-1) * value, dim=1)

    def forward(self, x_ctrl, dose_feat, drug_feat, gene_feat):
        z_cell = self.proj(self.cell_encoder(x_ctrl))
        z_drug = self.drug_encoder(drug_feat)
        z_dose = torch.sigmoid(self.dose_encoder(dose_feat))
        z_drug = self.proj(z_drug * z_dose)

        if gene_feat.dim() == 3:
            gene_feat = gene_feat[0]
        z_gene = self.proj(self.gene_encoder(gene_feat))

        z_drug_ctx = self.gene_attention(z_drug, z_gene)
        z_cell_ctx = self.gene_attention(z_cell, z_gene)

        batch_size, gene_count = x_ctrl.shape
        z_drug_expand = z_drug_ctx.unsqueeze(1).repeat(1, gene_count, 1)
        z_cell_expand = z_cell_ctx.unsqueeze(1).repeat(1, gene_count, 1)
        z_gene_expand = z_gene.unsqueeze(0).repeat(batch_size, 1, 1)

        hidden = torch.cat([z_drug_expand, z_gene_expand, z_cell_expand], dim=-1)
        y_pred = self.predictor(hidden).squeeze(-1)

        return {
            "y_pred": y_pred,
            "zc_ctrl": z_cell_ctx,
            "zd": z_drug_ctx,
        }
