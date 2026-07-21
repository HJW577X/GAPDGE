import torch
import torch.nn.functional as F


def info_nce(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def loss_total(pred_y, true_y, zc_ctrl, zd, lambda_cl=0.01, temp=0.1):
    pred_loss = F.mse_loss(pred_y, true_y)
    loss_cl = info_nce(zc_ctrl, zd, temperature=temp)
    total = pred_loss + lambda_cl * loss_cl
    return total, {"pred_loss": pred_loss.item(), "loss_cl": loss_cl.item()}
