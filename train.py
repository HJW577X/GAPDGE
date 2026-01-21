import os, logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import loss_total
from utils import save_checkpoint
import numpy as np
from scipy.stats import pearsonr

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, device, out_dir, split_key, cfg):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.split_key = split_key
        os.makedirs(self.out_dir + '/' + self.split_key, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_ckpt_path = self.out_dir + '/' + self.split_key + '/best_model_' + self.timestamp + '.pth'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.get('lr',1e-4), weight_decay=cfg.get('weight_decay',1e-8))
        self.scaler = torch.cuda.amp.GradScaler()
        self.epochs = cfg.get('epochs',50)
        self.lambda_cl = cfg.get('lambda',0.1)
        self.temp = cfg.get('temp',0.1)
        self.best_val = float('inf')
        self.best_epoch = -1
        self.history = {'train_loss':[], 'val_loss':[]}
        logging.basicConfig(
            filename = self.out_dir + '/' + self.split_key + '/' + 'train_' + self.timestamp + '.log',  
            level = logging.DEBUG,  
            format = '%(asctime)s - %(levelname)s - %(message)s'  
        )

    def train_epoch(self):
        self.model.train()
        losses=[]
        losses_mse=[]
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train')
        for i, batch in loop:
            y, x, dose, drug, gene = batch
            y = y.to(self.device).float()
            x = x.to(self.device).float()
            dose = dose.to(self.device).float()
            drug = drug.to(self.device).float()
            gene = gene.to(self.device).float()

            with torch.amp.autocast('cuda'):
                out = self.model(x, dose, drug, gene)
                y_pred = out['y_pred']
                zc_ctrl = out['zc_ctrl']
                zd = out['zd']
                total_loss, loss_dict = loss_total(y_pred, y, zc_ctrl, zd, lambda_cl=self.lambda_cl, temp=self.temp)

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(total_loss.item())
            losses_mse.append(loss_dict['pred_loss'])
            loop.set_postfix(loss=sum(losses)/len(losses))
        return float(sum(losses)/len(losses)), float(sum(losses_mse)/len(losses_mse))

    @torch.no_grad()
    def evaluate_val(self, loader):
        if loader is None:
            return None
        self.model.eval()
        losses=[]
        losses_mse=[]
        all_y_true = []  
        all_y_pred = []  
        loop = tqdm(enumerate(loader), total=len(loader), desc='Eval')
        for i, batch in loop:
            y, x, dose, drug, gene = batch
            y = y.to(self.device).float()
            x = x.to(self.device).float()
            dose = dose.to(self.device).float()
            drug = drug.to(self.device).float()
            gene = gene.to(self.device).float()
            out = self.model(x, dose, drug, gene)
            y_pred = out['y_pred']
            all_y_true.append(y.cpu().detach().numpy().ravel())
            all_y_pred.append(y_pred.cpu().detach().numpy().ravel())
            zc_ctrl = out['zc_ctrl']; zd = out['zd']
            total_loss, loss_dict = loss_total(y_pred, y, zc_ctrl, zd, lambda_cl=self.lambda_cl, temp=self.temp)
            losses.append(total_loss.item())
            losses_mse.append(loss_dict['pred_loss'])
            loop.set_postfix(val_loss=sum(losses)/len(losses))

        if all_y_true and all_y_pred:
            y_true = np.concatenate(all_y_true)
            y_pred = np.concatenate(all_y_pred)
            ss_res = np.sum((y_true - y_pred) ** 2) 
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0  
            pcc = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        else:
            r2 = 0.0
            pcc = 0.0
        return float(sum(losses)/len(losses)), float(sum(losses_mse)/len(losses_mse)), r2, pcc

    @torch.no_grad()
    def evaluate_test(self, loader):
        if loader is None:
            return None
        self.model.eval()
        losses = []
        losses_mse = []
        
        gene_set_data = {
            'all': {'true': [], 'pred': [], 'dac': []},
            'de50': {'true': [], 'pred': [], 'dac': []},
            'de100': {'true': [], 'pred': [], 'dac': []},
        }

 
        loop = tqdm(enumerate(loader), total=len(loader), desc='Eval')
        for i, batch in loop:
            y, x, dose, drug, gene = batch  
            y = y.to(self.device).float()
            x = x.to(self.device).float()
            dose = dose.to(self.device).float()
            drug = drug.to(self.device).float()
            gene = gene.to(self.device).float()
            out = self.model(x, dose, drug, gene)
            y_pred = out['y_pred'] 
            zc_ctrl = out['zc_ctrl']; zd = out['zd'] 
            total_loss, loss_dict = loss_total(
                y_pred, y, zc_ctrl, zd,  
                lambda_cl=self.lambda_cl, temp=self.temp
            )
            losses.append(total_loss.item())
            losses_mse.append(loss_dict['pred_loss'])
            loop.set_postfix(val_loss=sum(losses)/len(losses))
            y_true_np = y.cpu().detach().numpy()  
            y_pred_np = y_pred.cpu().detach().numpy() 
            x_true_np = x.cpu().detach().numpy()  
 
            for sample_idx in range(y_true_np.shape[0]):

                yi = y_true_np[sample_idx]  
                ypi = y_pred_np[sample_idx] 
                xi = x_true_np[sample_idx]  
 
                idx50 = self.get_deg_indices(xi, yi, top_k=50)
                idx100 = self.get_deg_indices(xi, yi, top_k=100)
 
                # delta
                delta_true = yi - xi
                delta_pred = ypi - xi

                # All 
                gene_set_data['all']['true'].append(yi.ravel())
                gene_set_data['all']['pred'].append(ypi.ravel())
                gene_set_data['all']['dac'].append(self.directional_accuracy_delta(delta_true, delta_pred))
                # DE50
                gene_set_data['de50']['true'].append(delta_true[idx50].ravel())
                gene_set_data['de50']['pred'].append(delta_pred[idx50].ravel())
                gene_set_data['de50']['dac'].append(self.directional_accuracy_delta(delta_true[idx50], delta_pred[idx50]))
                # DE100
                gene_set_data['de100']['true'].append(delta_true[idx100].ravel())
                gene_set_data['de100']['pred'].append(delta_pred[idx100].ravel())
                gene_set_data['de100']['dac'].append(self.directional_accuracy_delta(delta_true[idx100], delta_pred[idx100]))


        metrics = {}
        for gene_set in gene_set_data.keys():
            true_data = gene_set_data[gene_set]['true']
            pred_data = gene_set_data[gene_set]['pred']
            
            if not true_data or not pred_data:
                metrics[f'r2_{gene_set}'] = 0.0
                metrics[f'pcc_{gene_set}'] = 0.0

                continue
 
            y_true_flat = np.concatenate(true_data)
            y_pred_flat = np.concatenate(pred_data)
 
            ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
            ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
 
            pcc = pearsonr(y_true_flat, y_pred_flat)[0] if len(y_true_flat) > 1 else 0.0
            pcc = pcc if not np.isnan(pcc) else 0.0
 
            metrics[f'r2_{gene_set}'] = r2
            metrics[f'pcc_{gene_set}'] = pcc
            metrics[f'dac_{gene_set}'] = float(np.mean(gene_set_data[gene_set]['dac']))
 
        avg_loss = float(sum(losses)/len(losses)) if losses else 0.0
        avg_mse = float(sum(losses_mse)/len(losses_mse)) if losses_mse else 0.0
 
        return (
            avg_loss, avg_mse,
            metrics['r2_all'], metrics['r2_de50'], metrics['r2_de100'],
            metrics['pcc_all'], metrics['pcc_de50'], metrics['pcc_de100'],
            metrics['dac_all'], metrics['dac_de50'], metrics['dac_de100'],
        )
 
    # def get_deg_indices(self, x_ctrl, y_pert, top_k=50):
    #     logfc = np.log2((y_pert + 1e-8) / (x_ctrl + 1e-8))
    #     sorted_indices = np.argsort(np.abs(logfc))[::-1][:top_k]
    #     return sorted_indices
    def directional_accuracy_delta(self, delta_true, delta_pred):
        """
        delta_true, delta_pred: (K,) numpy arrays
        """
        sign_true = np.sign(delta_true)
        sign_pred = np.sign(delta_pred)

        return np.mean(sign_true == sign_pred)


    def get_deg_indices(self, x_ctrl, y_pert, top_k):
        delta = np.abs(y_pert - x_ctrl)
        return np.argsort(delta)[::-1][:top_k]

    def fit(self):
        for ep in range(self.epochs):
            print(f'Epoch {ep} start {datetime.now()}')
            logging.info(f'Epoch {ep} start {datetime.now()}')
            train_loss, train_mse = self.train_epoch()
            val_loss, val_mse, val_r2, val_pcc = self.evaluate_val(self.valid_loader) if self.valid_loader is not None else None
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            print(f'Epoch {ep}: train_loss={train_loss}, train_mse={train_mse}, val_loss={val_loss}, val_mse={val_mse}, val_r2={val_r2}, val_pcc={val_pcc}')
            logging.info(f'Epoch {ep}: train_loss={train_loss}, train_mse={train_mse}, val_loss={val_loss}, val_mse={val_mse}, val_r2={val_r2}, val_pcc={val_pcc}')

            test_loss, test_mse, test_r2_all, test_r2_de50, test_r2_de100, test_pcc_all, test_pcc_de50, test_pcc_de100, test_dac_all, test_dac_de50, test_dac_de100 = self.evaluate_test(self.test_loader) if self.test_loader is not None else None
            print(f'Epoch {ep}:')
            print(f'  Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}')
            print(f'  Test R2 (all/de50/de100): {test_r2_all:.4f}/{test_r2_de50:.4f}/{test_r2_de100:.4f}')
            print(f'  Test PCC (all/de50/de100): {test_pcc_all:.4f}/{test_pcc_de50:.4f}/{test_pcc_de100:.4f}')
            print(f'  Test DAC (all/de50/de100): {test_dac_all:.4f}/{test_dac_de50:.4f}/{test_dac_de100:.4f}')
            logging.info(f'Epoch {ep}:')
            logging.info(f'  Test Loss: {test_loss:.4f}, Test MSE: {test_mse:.4f}')
            logging.info(f'  Test R2 (all/de50/de100): {test_r2_all:.4f}/{test_r2_de50:.4f}/{test_r2_de100:.4f}')
            logging.info(f'  Test PCC (all/de50/de100): {test_pcc_all:.4f}/{test_pcc_de50:.4f}/{test_pcc_de100:.4f}')
            logging.info(f'  Test DAC (all/de50/de100): {test_dac_all:.4f}/{test_dac_de50:.4f}/{test_dac_de100:.4f}')
            
            if val_loss is not None and val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = ep
                save_checkpoint({'epoch':ep, 'model_state':self.model.state_dict()}, self.model_ckpt_path)
            if ep - self.best_epoch > 10 and ep > 20:
                print('Early stopping triggered')
                logging.info('Early stopping triggered')
                break
        ckpt = self.model_ckpt_path
        if os.path.exists(ckpt):
            ck = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(ck['model_state'])
        return self.history
