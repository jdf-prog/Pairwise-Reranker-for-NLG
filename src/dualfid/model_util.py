import torch
import torch.nn.functional as F
def regression_BCE_loss(x, aux_loss, scores):

    scores = scores.to(x.device)
    assert x.shape == scores.shape
    # compute contrastive loss
    if aux_loss is not None:
        loss = torch.tensor(aux_loss).to(x.device)
    else:
        loss = torch.tensor(0.0).to(x.device)
    labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(x.device)
    loss = F.binary_cross_entropy(x, labels, reduction='mean')
    return torch.sum(x, dim=-1), loss
