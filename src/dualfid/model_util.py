import torch
import os
import numpy as np
import torch.nn.functional as F
import transformers
from dualfid.data import Dataset
from transformers import TrainingArguments, Trainer, WEIGHTS_NAME, CONFIG_NAME
from dualfid.reranker import (
    SCR,
    DualReranker,
    CrossCompareReranker,
    CompareGenReranker,
    DualCompareReranker
)
from dualfid.fid import (
    DualFiDBART,
    DualFiDT5,
    FiDBART,
    FiDT5,
)
from dualfid.collator import (
    DualCollator,
    DualCompareCollator,
    DualFiDCollator,
    FiDCollator,
    SCRCollator,
    CrossCompareCollator,
    CompareGenCollator
)
from transformers import (
    RobertaModel,
    BertModel,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
)
from transformers.models.roberta.modeling_roberta import RobertaModel


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
    elif model_type.startswith("bart"):
        print("\nUsing BART model")
        model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir = cache_dir)
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
    elif model_type.startswith("bart"):
        print("\nUsing BART tokenizer")
        from transformers import BartTokenizerFast
        tokenizer = BartTokenizerFast.from_pretrained(model_name, cache_dir = cache_dir)
    return tokenizer

def build_reranker(reranker_type, model_type, model_name, cache_dir, config):
    reranker = None
    pretrained_model = build_pretrained_model(model_type, model_name, cache_dir)

    if reranker_type == "scr":
        reranker = SCR(pretrained_model, config)
    elif reranker_type == "dual":
        reranker = DualReranker(pretrained_model, config)
    elif reranker_type == "crosscompare":
        reranker = CrossCompareReranker(pretrained_model, config)
    elif reranker_type == "dualcompare":
        reranker = DualCompareReranker(pretrained_model, config)
    elif reranker_type == "comparegen":
        reranker = CompareGenReranker(pretrained_model, config)

    return reranker

def build_fid(fid_type, model_type, model_name, cache_dir, config):
    fid = None
    if fid_type == 'fid':
        if model_type.startswith("t5"):
            fid_class = FiDT5
        elif model_type.startswith("bart"):
            fid_class = FiDBART
        else:
            raise ValueError(f"model_type {model_type} not supported")
    elif fid_type == "dualfid":
        if model_type.startswith("t5"):
            fid_class = DualFiDT5
        elif model_type.startswith("bart"):
            fid_class = DualFiDBART
        else:
            raise ValueError(f"model_type {model_type} not supported")
    else:
        raise NotImplementedError

    hf_model = build_pretrained_model(model_type, model_name, cache_dir)
    hf_config = hf_model.config
    for k, v in config.items():
        setattr(hf_config, k, v)
    fid = fid_class(hf_model.config)
    fid.load_hfm(hf_model.state_dict())

    return fid

def build_collator(model_type:str):
    if model_type == "fid":
        return FiDCollator
    elif model_type == "dualfid":
        return DualFiDCollator
    elif model_type == "scr":
        return SCRCollator
    elif model_type == "dual":
        return DualCollator
    elif model_type == "crosscompare":
        return CrossCompareCollator
    elif model_type == "dualcompare":
        return DualCompareCollator
    elif model_type == "t5comparegen":
        return CompareGenCollator
    else:
        raise ValueError(f"model_type {model_type} not supported")
