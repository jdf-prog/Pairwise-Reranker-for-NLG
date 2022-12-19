import torch
import os
import numpy as np
import torch.nn.functional as F
from reranker.reranker import (
    SCR,
    DualReranker,
    CrossCompareReranker,
    DualCompareReranker
)
from reranker.fid import (
    DualFiDBART,
    DualFiDT5,
    FiDBART,
    FiDT5,
)
from reranker.collator import (
    DualCollator,
    DualCompareCollator,
    DualFiDCollator,
    FiDCollator,
    SCRCollator,
    CrossCompareCollator,
    CurriculumCrossCompareCollator,
)
from transformers import (
    RobertaModel,
    BertModel,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
)
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.deberta.modeling_deberta import DebertaModel


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
    elif model_type.startswith("deberta"):
        print("\nUsing DeBERTa model")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, cache_dir = cache_dir)
    elif model_type.startswith("xlm-roberta"):
        print("\nUsing XLM-RoBERTa model")
        from transformers import XLMRobertaModel
        model = XLMRobertaModel.from_pretrained(model_name, cache_dir = cache_dir)
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
    elif model_type.startswith("deberta"):
        print("\nUsing DeBERTa tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    elif model_type.startswith("xlm-roberta"):
        print("\nUsing XLM-RoBERTa tokenizer")
        from transformers import XLMRobertaTokenizerFast
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name, cache_dir = cache_dir)
    return tokenizer

def build_reranker(reranker_type, model_type, model_name, cache_dir, config, tokenizer=None):
    reranker = None
    pretrained_model = build_pretrained_model(model_type, model_name, cache_dir)
    pretrained_model.resize_token_embeddings(config["new_num_tokens"])

    if reranker_type == "scr":
        reranker = SCR(pretrained_model, config, tokenizer)
    elif reranker_type == "dual":
        reranker = DualReranker(pretrained_model, config, tokenizer)
    elif reranker_type == "crosscompare":
        reranker = CrossCompareReranker(pretrained_model, config, tokenizer)
    elif reranker_type == "dualcompare":
        reranker = DualCompareReranker(pretrained_model, config, tokenizer)

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

def build_collator(
    model_type:str,
    tokenizer,
    source_max_length:int,
    candidate_max_length:int,
    source_prefix:str = None,
    candidate1_prefix:str = None,
    candidate2_prefix:str = None,
    curriculum_learning:bool = False,
    ):
    if model_type == "fid":
        return FiDCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix)
    elif model_type == "dualfid":
        return DualFiDCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix)
    elif model_type == "scr":
        return SCRCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix)
    elif model_type == "dual":
        return DualCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix)
    elif model_type == "crosscompare":
        if curriculum_learning:
            return CurriculumCrossCompareCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix, candidate2_prefix)
        else:
            return CrossCompareCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix, candidate2_prefix)
    elif model_type == "dualcompare":
        return DualCompareCollator(source_max_length, tokenizer, candidate_max_length, source_prefix, candidate1_prefix)
    else:
        raise ValueError(f"model_type {model_type} not supported")


