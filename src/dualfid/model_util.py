import torch
import os
import numpy as np
import torch.nn.functional as F
from dualfid.data import Dataset
from transformers import TrainingArguments, Trainer, WEIGHTS_NAME, CONFIG_NAME
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

def build_reranker_from_checkpoint(reranker_type, model_type, model_name, cache_dir, checkpoint):
    config = torch.load(os.path.join(checkpoint, "config.bin"))
    reranker = build_reranker(reranker_type, model_type, model_name, cache_dir, config)
    reranker.load_state_dict(torch.load(os.path.join(checkpoint, "pytorch_model.bin")))
    training_args = torch.load(os.path.join(checkpoint, "training_args.bin"))
    optimizer = torch.load(os.path.join(checkpoint, "optimizer.pt"))
    scheduler = torch.load(os.path.join(checkpoint, "scheduler.pt"))
    return reranker, training_args, optimizer, scheduler, config

def save_reranker_checkpoint(trainer, reranker, save_path, checkpoint_name):
    checkpoint_dir = os.path.join(save_path, checkpoint_name)
    # save training args and model parameters
    trainer.save_model(checkpoint_dir)
    # save optimizer and scheduler
    torch.save(trainer.optimizer, os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(trainer.lr_scheduler, os.path.join(checkpoint_dir, "scheduler.pt"))
    # save config
    torch.save(reranker.config, os.path.join(checkpoint_dir, "config.bin"))

