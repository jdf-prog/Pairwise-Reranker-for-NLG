import torch
import torch.nn.functional as F
from dualfid.data import Dataset
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

def augment_training_data(dataset: Dataset):
    # argument data
    augment_data = []
    for item in dataset.data:
        max_score_candidate = sorted(item['candidates'], key=lambda x: sum(x['scores'].values()), reverse=True)[:2]
        augment_data.append(item)
        for candidate in max_score_candidate:
            augment_item = item.copy()
            augment_item['target'] = candidate['text']
            augment_data.append(augment_item)
    dataset.data = augment_data
