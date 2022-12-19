"""
    This file is taken from This file is modified based on:
    https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/candidate_generation/engine.py
    We thank the authors for sharing their code.
"""
import gc

from tqdm import tqdm

import torch

def get_candidates(tokenizer, val_loader, model, device, args, **kwargs):
    val_texts = []
    val_candidates = []
    val_labels = []
    base_model = model.pretrained_model

    for idx, batch in tqdm(enumerate(val_loader)):
        for k in batch["source_inputs"].keys():
            batch["source_inputs"][k] = batch["source_inputs"][k].to(device)
            if len(batch["source_inputs"][k].shape) > 2:
                batch["source_inputs"][k] = batch["source_inputs"][k].squeeze(1)

        model.zero_grad()
        val_texts += batch["source"]

        raw_candidates = beam_search_step(batch, tokenizer, base_model, device, args, **kwargs)

        candidates = []
        for i in range(len(batch["source"])):
            candidates.append(raw_candidates[i*args.num_return_sequences:(i+1)*args.num_return_sequences])
        val_candidates += candidates

        labels = batch["target"]
        val_labels += labels

    print(len(val_texts), len(val_candidates), len(val_candidates[0]), len(val_labels))

    return val_texts, val_candidates, val_labels


def beam_search_step(batch, tokenizer, base_model, device, args, **kwargs):
    # 1 - beam search
    if args.generation_method == "beam_search":
        summary_ids = base_model.generate(
            batch["source_inputs"]['input_ids'],
            attention_mask = batch["source_inputs"]["attention_mask"],
            num_beams = args.num_beams,
            num_return_sequences = args.num_return_sequences,
            max_length = args.candidate_max_length,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True,
            **kwargs
        )
    # 2 - diverse beam search
    if args.generation_method == "diverse_beam_search":
        summary_ids = base_model.generate(
            batch["source_inputs"]['input_ids'],
            attention_mask = batch["source_inputs"]["attention_mask"],
            num_beams = args.num_beams,
            num_beam_groups = args.num_beam_groups,
            num_return_sequences = args.num_return_sequences,
            max_length = args.candidate_max_length,
            diversity_penalty = args.diversity_penalty,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True,
            **kwargs
        )
    # 3 - top-p sampling
    if args.generation_method == "top_p_sampling":
        summary_ids = base_model.generate(
            batch["source_inputs"]['input_ids'],
            attention_mask = batch["source_inputs"]["attention_mask"],
            num_beams = 1,
            do_sample = True,
            top_p = args.top_p,
            num_return_sequences = args.num_return_sequences,
            max_length = args.candidate_max_length,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True,
            **kwargs
        )
    # 4 - top-k sampling
    if args.generation_method == "top_k_sampling":
        summary_ids = base_model.generate(
            batch["source_inputs"]['input_ids'],
            attention_mask = batch["source_inputs"]["attention_mask"],
            num_beams = 1,
            do_sample = True,
            top_k = args.top_k,
            num_return_sequences = args.num_return_sequences,
            max_length = args.candidate_max_length,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True,
            **kwargs
        )
    generated = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    del summary_ids
    gc.collect()

    return generated
