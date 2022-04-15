import torch
import torch.nn.functional as F

def compute_SimCSE_loss(y_pred, tao=0.05, device="cuda"):
    idxes = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxes + 1 - idxes % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)