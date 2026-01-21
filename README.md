# Gene-Aware Drug Perturbed Transcriptional Responses Prediction (GAPDGE)
Accurately predicting drug perturbed gene expression is essential for drug discovery and biological insight. However, current methods either treat genes as isolated statistical dimensions, neglecting their intrinsic structural and functional priors, or treat drugs and cells as independent modalities, limiting their generalizability to unseen perturbational conditions. To address these limitations, we leverage genomic and chemical foundation models to extract rich biological priors and generate foundational gene and drug representations. Building on this, we introduce a gene-aware attention mechanism that employs these representations as queries and keys to adaptively modulate the impact of perturbations based on cellular state, thereby capturing gene-specific response patterns. Furthermore, a contrastive alignment objective is utilized to synchronize drug and cellular representations in a shared latent space, enabling robust prediction of drug responses to unseen conditions.

## Setup the environment with Anaconda
You can create a new Python environment:
```
conda create -n GAPDGE python=3.9
```
Then, you can activate the environment using:
```
conda activate GAPDGE
pip install -r requirements.txt
```

## Download data
You can also directly download files from: https://cloud.tsinghua.edu.cn/f/7bca2e22c1f14c4db7db/?dl=1.

Then put lincs_adata.h5ad, pert_smiles_emb.pkl, dosage_prompt_emb_lincs.pkl into the data folder.


## Train and test
You can directly run the model using the default parameters via:
```
python main.py
```
