import torch
import numpy as np
import torch.nn.functional as F
from dualfid.data import Dataset
from dualfid.reranker import (
    SCR,
    T5SCR,
    DualReranker
)
from transformers import (
    RobertaModel,
    BertModel,
    T5ForConditionalGeneration
)
def regression_BCE_loss(x, aux_loss, scores):

    scores = scores.to(x.device)
    assert x.shape == scores.shape
    # compute contrastive loss
    if aux_loss is not None:
        loss = torch.tensor(aux_loss).to(x.device)
    else:
        loss = torch.tensor(0.0).to(x.device)
    # labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(x.device) # only the best one
    labels = torch.gt(scores, torch.mean(scores, dim=1, keepdim=True)[0]).float().to(x.device) # select half of the best ones
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

def build_pretrained_model(model_type, model_name, cache_dir=None):
    model = None
    if model_type.startswith("roberta"):
        print("\nUsing RoBERTa model")
        model = RobertaModel.from_pretrained(model_name, cache_dir = cache_dir)
    elif model_type.startswith("bert"):
        print("\nUsing BERT model")
        model = BertModel.from_pretrained(model_name, cache_dir = cache_dir)
    elif model_type.startswith("t5"):
        print("\nUsing T5 model")
        model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir = cache_dir)
    return model

def build_tokenizer(model_type, model_name, cache_dir=None):
    tokenizer = None
    if model_type.startswith("roberta"):
        print("\nUsing RoBERTa tokenizer")
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir = cache_dir)
    elif model_type.startswith("bert"):
        print("\nUsing BERT tokenizer")
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir = cache_dir)
    elif model_type.startswith("t5"):
        print("\nUsing T5 tokenizer")
        from transformers import T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained(model_name, cache_dir = cache_dir)
    return tokenizer

def build_reranker(reranker_type, model_type, model_name, cache_dir, config):
    reranker = None
    pretrained_model = build_pretrained_model(model_type, model_name, cache_dir)

    if reranker_type == "scr":
        if model_type.startswith("t5"):
            reranker = T5SCR(pretrained_model, config)
        else:
            reranker = SCR(pretrained_model, config)
    elif reranker_type == "dual":
        reranker = DualReranker(pretrained_model, config)
    return reranker

def build_reranker_from_checkpoint(reranker_type, model_type, model_name, cache_dir, checkpoint, **kwargs):
    reranker = build_reranker(reranker_type, model_type, model_name, cache_dir, **kwargs)
    reranker.load_state_dict(torch.load(checkpoint))
    return reranker

def sub_sampling(input_ids, attention_mask, scores, num_pos, num_neg, mode="bottom"):
    batch_size, n_candidate, seq_len = input_ids.shape
    selected_idx = []
    for i in range(batch_size):
        idx = np.arange(n_candidate)
        # remove duplicate candidates, cpu
        unique_idx = []
        unique_scores = []
        for j, score in enumerate(torch.sum(scores[i], dim=-1)):
            if score not in unique_scores:
                unique_idx.append(idx[j])
                unique_scores.append(score.item())
        unique_idx = np.array(unique_idx)
        unique_scores = np.array(unique_scores)
        # only select a few pos and neg candidates
        sorted_idx = np.argsort(unique_scores)[::-1]
        pos_idx = sorted_idx[:num_pos]
        if mode == "bottom":
            neg_idx = sorted_idx[-num_neg:]
        elif mode == "random":
            neg_idx = np.random.choice(sorted_idx[num_pos:], num_neg, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        idx = unique_idx[idx]
        selected_idx.append(idx)
    selected_idx = torch.tensor(selected_idx).to(input_ids.device)
    input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
    attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
    scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
    return input_ids, attention_mask, scores
