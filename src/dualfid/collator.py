from sklearn.utils import shuffle
import torch
import random
import json
import numpy as np

def encode_texts(texts, tokenizer, max_length):
    """
    Args:
        texts List[str]: [n_texts]
    Returns:
        input_ids: [n_texts, max_length]
        attention_mask: [n_texts, max_length]
    """
    p = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    return p['input_ids'], p['attention_mask'].bool()

def encode_batch_text(batch_texts, tokenizer, max_length):
    """
    Args:
        batch_texts List[str]: [batch_size, n_texts]
    Returns:
        batch_input_ids: [batch_size, n_texts, max_length]
        batch_attention_mask: [batch_size, n_texts, max_length]
    """
    encoded_ids, encoded_masks = [], []
    for k, texts in enumerate(batch_texts):
        if isinstance(texts, str):
            texts = [texts]
        ids, mask = encode_texts(texts, tokenizer, max_length)
        encoded_ids.append(ids[None])
        encoded_masks.append(mask[None])
    encoded_ids = torch.cat(encoded_ids, dim=0)
    encoded_masks = torch.cat(encoded_masks, dim=0)
    return encoded_ids, encoded_masks.bool()

def get_truncated_text(texts, tokenizer, max_length):
    """
        Truncate the texts to max_length
    """
    truncated_texts = []
    for text in texts:
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )
        truncated_texts.append(tokenizer.decode(tokens, skip_special_tokens=True))
    return truncated_texts

class SCRCollator(object):
    def __init__(self, source_maxlength, tokenizer, candidate_maxlength, postfix=None):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.sep_token = "[SEP]" # debug
        self.separate_token = self.sep_token
        self.postfix = postfix
        self.target_maxlength = self.candidate_maxlength


    def __call__(self, batch):
        batch_size = len(batch)
        batch_source = [b['source'] for b in batch]
        batch_target = [b['target'] for b in batch]
        batch_candidates = [b['candidates'] for b in batch]
        batch_scores = [b['scores'] for b in batch]

        batch_source = get_truncated_text(batch_source, self.tokenizer, self.source_maxlength)
        batch_candidates = [get_truncated_text(c, self.tokenizer, self.candidate_maxlength) for c in batch_candidates]
        batch_target = get_truncated_text(batch_target, self.tokenizer, self.target_maxlength)

        if self.postfix is not None:
            source_texts = [[self.separate_token.join([s, c, self.postfix]) for c in cands] for s, cands in zip(batch_source, batch_candidates)] # concatenate source and candidate
            target_texts = [self.separate_token.join([s, t, self.postfix]) for s, t in zip(batch_source, batch_target)]
        else:
            source_texts = [[self.separate_token.join([s, c]) for c in cands] for s, cands in zip(batch_source, batch_candidates)] # concatenate source and target
            target_texts = [self.separate_token.join([s, t]) for s, t in zip(batch_source, batch_target)]

        encoded_source_text_ids, encoded_source_text_masks = encode_batch_text(source_texts, self.tokenizer, self.tokenizer.model_max_length) # source
        encoded_target_text_ids, encoded_target_text_masks = encode_texts(target_texts, self.tokenizer, self.tokenizer.model_max_length) # target


        return {
            'input_ids' : encoded_source_text_ids,
            'attention_mask' : encoded_source_text_masks,
            'target_ids' : encoded_target_text_ids,
            'target_attention_mask' : encoded_target_text_masks,
            'scores' : torch.stack(batch_scores, dim=0),
        }

class DualCollator(object):
    def __init__(self, source_maxlength, tokenizer, candidate_maxlength):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.cls_token = self.cls_token if self.cls_token is not None else ""
        self.separate_token = self.sep_token + ' ' + self.cls_token # used to separate 2 concatenated texts
        self.target_maxlength = self.candidate_maxlength


    def __call__(self, batch):
        batch_size = len(batch)
        batch_source = [b['source'] for b in batch]
        batch_target = [b['target'] for b in batch]
        batch_candidates = [b['candidates'] for b in batch]
        batch_scores = [b['scores'] for b in batch]

        batch_source = get_truncated_text(batch_source, self.tokenizer, self.source_maxlength)
        batch_candidates = [get_truncated_text(c, self.tokenizer, self.candidate_maxlength) for c in batch_candidates]
        batch_target = get_truncated_text(batch_target, self.tokenizer, self.target_maxlength)

        encoded_source_ids, encoded_source_masks = encode_texts(batch_source, self.tokenizer, self.tokenizer.model_max_length) # source
        encoded_target_ids, encoded_target_masks = encode_texts(batch_target, self.tokenizer, self.tokenizer.model_max_length) # target
        encoded_candidate_ids, encoded_candidate_masks = encode_batch_text(batch_candidates, self.tokenizer, self.tokenizer.model_max_length) # candidates

        return {
            'source_ids' : encoded_source_ids,
            'source_attention_mask' : encoded_source_masks,
            'target_ids' : encoded_target_ids,
            'target_attention_mask' : encoded_target_masks,
            "candidate_ids" : encoded_candidate_ids,
            "candidate_attention_mask" : encoded_candidate_masks,
            'scores' : torch.stack(batch_scores, dim=0),
        }

class CrossCompareCollator(object):
    def __init__(self, source_maxlength, tokenizer, candidate_maxlength, postfix=None):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.separate_token = self.sep_token
        self.postfix = postfix
        self.target_maxlength = self.candidate_maxlength


    def __call__(self, batch):
        batch_size = len(batch)
        batch_source = [b['source'] for b in batch]
        batch_target = [b['target'] for b in batch]
        batch_candidates = [b['candidates'] for b in batch]
        batch_scores = [b['scores'] for b in batch]
        batch_source = get_truncated_text(batch_source, self.tokenizer, self.source_maxlength)
        batch_candidates = [get_truncated_text(c, self.tokenizer, self.candidate_maxlength) for c in batch_candidates]
        n_candidate = len(batch_candidates[0])

        batch_candidate_pairs = [[[None for _ in range(n_candidate)] for _ in range(n_candidate)] for _ in range(batch_size)]
        batch_cand_target_pairs = [[None for _ in range(n_candidate)] for _ in range(batch_size)]

        target_rand_mat = torch.rand(batch_size, n_candidate) > 0.5
        scores = torch.stack(batch_scores, dim=0)
        batch_cand_target_dif_scores = torch.where(target_rand_mat.unsqueeze(-1), scores - 1.0, 1.0 - scores)

        for bz in range(batch_size):
            n_candidate = len(batch_candidates[bz])
            for i in range(n_candidate):
                for j in range(n_candidate):
                    if self.postfix is not None:
                        batch_candidate_pairs[bz][i][j] = self.separate_token.join([batch_source[bz], batch_candidates[bz][i], batch_candidates[bz][j], self.postfix])
                    else:
                        batch_candidate_pairs[bz][i][j] = self.separate_token.join([batch_source[bz], batch_candidates[bz][i], batch_candidates[bz][j]])
                if target_rand_mat[bz][i]:
                    batch_cand_target_pairs[bz][i] = self.separate_token.join([batch_candidates[bz][i], batch_target[bz]])
                else:
                    batch_cand_target_pairs[bz][i] = self.separate_token.join([batch_target[bz], batch_candidates[bz][i]])

        encoded_source_ids, encoded_source_masks = encode_texts(batch_source, self.tokenizer, self.tokenizer.model_max_length) # source
        encoded_cand_target_ids, encoded_cand_target_masks = encode_batch_text(batch_cand_target_pairs, self.tokenizer, self.tokenizer.model_max_length) # candidates
        encoded_cand_pair_ids, encoded_cand_pair_masks = [], []
        for bz in range(batch_size):
            n_candidate = len(batch_candidates[bz])
            ids, mask = encode_batch_text(batch_candidate_pairs[bz], self.tokenizer, self.tokenizer.model_max_length)
            encoded_cand_pair_ids.append(ids)
            encoded_cand_pair_masks.append(mask)
        encoded_cand_pair_ids = torch.stack(encoded_cand_pair_ids, dim=0)
        encoded_cand_pair_masks = torch.stack(encoded_cand_pair_masks, dim=0)


        return {
            'source_ids' : encoded_source_ids,
            'source_attention_mask' : encoded_source_masks,
            "candidate_pair_ids" : encoded_cand_pair_ids,
            "candidate_pair_attention_mask" : encoded_cand_pair_masks,
            "candidate_target_ids" : encoded_cand_target_ids,
            "candidate_target_attention_mask" : encoded_cand_target_masks,
            "scores" : scores,
            "cand_target_dif_scores" : batch_cand_target_dif_scores,
        }

class DualCompareCollator(object):
    def __init__(self, source_maxlength, tokenizer, candidate_maxlength, postfix=None):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.separate_token = self.sep_token
        self.postfix = postfix
        self.target_maxlength = self.candidate_maxlength


    def __call__(self, batch):
        batch_size = len(batch)
        batch_source = [b['source'] for b in batch]
        batch_target = [b['target'] for b in batch]
        batch_candidates = [b['candidates'] for b in batch]
        batch_scores = [b['scores'] for b in batch]
        batch_source = get_truncated_text(batch_source, self.tokenizer, self.source_maxlength)
        batch_candidates = [get_truncated_text(c, self.tokenizer, self.candidate_maxlength) for c in batch_candidates]
        n_candidate = len(batch_candidates[0])

        batch_candidate_pairs = [[[None for _ in range(n_candidate)] for _ in range(n_candidate)] for _ in range(batch_size)]
        batch_cand_target_pairs = [[None for _ in range(n_candidate)] for _ in range(batch_size)]

        target_rand_mat = torch.rand(batch_size, n_candidate) > 0.5
        scores = torch.stack(batch_scores, dim=0)
        batch_cand_target_dif_scores = torch.where(target_rand_mat.unsqueeze(-1), scores - 1.0, 1.0 - scores)

        for bz in range(batch_size):
            n_candidate = len(batch_candidates[bz])
            for i in range(n_candidate):
                for j in range(n_candidate):
                    if self.postfix is not None:
                        batch_candidate_pairs[bz][i][j] = self.separate_token.join([batch_candidates[bz][i], batch_candidates[bz][j], self.postfix])
                    else:
                        batch_candidate_pairs[bz][i][j] = self.separate_token.join([batch_candidates[bz][i], batch_candidates[bz][j]])
                if target_rand_mat[bz][i]:
                    batch_cand_target_pairs[bz][i] = self.separate_token.join([batch_candidates[bz][i], batch_target[bz]])
                else:
                    batch_cand_target_pairs[bz][i] = self.separate_token.join([batch_target[bz], batch_candidates[bz][i]])

        encoded_source_ids, encoded_source_masks = encode_texts(batch_source, self.tokenizer, self.tokenizer.model_max_length) # source
        encoded_cand_target_ids, encoded_cand_target_masks = encode_batch_text(batch_cand_target_pairs, self.tokenizer, self.tokenizer.model_max_length) # candidates
        encoded_cand_pair_ids, encoded_cand_pair_masks = [], []
        for bz in range(batch_size):
            n_candidate = len(batch_candidates[bz])
            ids, mask = encode_batch_text(batch_candidate_pairs[bz], self.tokenizer, self.tokenizer.model_max_length)
            encoded_cand_pair_ids.append(ids)
            encoded_cand_pair_masks.append(mask)
        encoded_cand_pair_ids = torch.stack(encoded_cand_pair_ids, dim=0)
        encoded_cand_pair_masks = torch.stack(encoded_cand_pair_masks, dim=0)


        return {
            'source_ids' : encoded_source_ids,
            'source_attention_mask' : encoded_source_masks,
            "candidate_pair_ids" : encoded_cand_pair_ids,
            "candidate_pair_attention_mask" : encoded_cand_pair_masks,
            "candidate_target_ids" : encoded_cand_target_ids,
            "candidate_target_attention_mask" : encoded_cand_target_masks,
            "scores" : scores,
            "cand_target_dif_scores" : batch_cand_target_dif_scores,
        }


class CompareGenCollator(object):
    def __init__(self, source_maxlength, tokenizer, candidate_maxlength, postfix=None):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.separate_token = self.sep_token
        self.postfix = postfix
        self.target_maxlength = self.candidate_maxlength


    def __call__(self, batch):
        batch_size = len(batch)
        batch_source = [b['source'] for b in batch]
        batch_target = [b['target'] for b in batch]
        batch_candidates = [b['candidates'] for b in batch]
        batch_scores = [b['scores'] for b in batch]
        batch_source = get_truncated_text(batch_source, self.tokenizer, self.source_maxlength)
        batch_candidates = [get_truncated_text(c, self.tokenizer, self.candidate_maxlength) for c in batch_candidates]
        n_candidate = len(batch_candidates[0])

        scores = torch.stack(batch_scores, dim=0)
        sum_scores = torch.sum(scores, dim=-1)
        random_idxs1 = torch.random.randperm(n_candidate)
        random_idxs2 = random_idxs1.roll(1)
        if self.postfix is not None:
            batch_candidate_pairs = [
                [self.separate_token.join(s, cands[i], cands[j], self.postfix) for i, j in zip(random_idxs1, random_idxs2)]
                for s, cands in zip(batch_source, batch_candidates)
            ]
            batch_candidate_pair_targets = [
                [cands[i] if sum_scores[bz][i] > sum_scores[bz][j] else cands[j] for i, j in zip(random_idxs1, random_idxs2)]
                for bz, cands in enumerate(batch_candidates)
            ]
            batch_cand_target_pairs = [
                [self.separate_token.join(cands[i], t, self.postfix) for i in range(n_candidate)]
                for cands, t in zip(batch_candidates, batch_target)
            ]
            batch_cand_target_pair_targets = batch_target
        else:
            batch_candidate_pairs = [
                [self.separate_token.join(s, cands[i], cands[j]) for i, j in zip(random_idxs1, random_idxs2)]
                for s, cands in zip(batch_source, batch_candidates)
            ]
            batch_candidate_pair_targets = [
                [cands[i] if sum_scores[bz][i] > sum_scores[bz][j] else cands[j] for i, j in zip(random_idxs1, random_idxs2)]
                for bz, cands in enumerate(batch_candidates)
            ]
            batch_cand_target_pairs = [
                [self.separate_token.join(cands[i], t) for i in range(n_candidate)]
                for cands, t in zip(batch_candidates, batch_target)
            ]
            batch_cand_target_pair_targets = batch_target

        batch_candidate_pairs = np.array(batch_candidate_pairs)
        batch_candidate_pair_targets = np.array(batch_candidate_pair_targets)
        batch_cand_target_pairs = np.array(batch_cand_target_pairs)
        batch_cand_target_pair_targets = np.array(batch_cand_target_pair_targets)
        input_text = np.concatenate([batch_candidate_pairs, batch_cand_target_pairs], axis=1)
        target_text = np.concatenate([batch_candidate_pair_targets, batch_cand_target_pair_targets], axis=1)

        encoded_input_ids, encoded_input_masks = encode_batch_text(input_text, self.tokenizer, self.tokenizer.model_max_length)
        encoded_target_ids, encoded_target_masks = encode_batch_text(target_text, self.tokenizer, self.tokenizer.model_max_length)
        return {
            "input_ids" : encoded_input_ids,
            "attention_mask" : encoded_input_masks,
            "labels" : encoded_target_ids,
        }



class DualFiDCollator(object):
    def __init__(self, source_maxlength, tokenizer, candidate_maxlength):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        # encode target
        target = self.tokenizer.batch_encode_plus(
            target,
            padding="longest",
            return_tensors='pt',
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)
        # encode source text
        source = [ex['source'] for ex in batch]
        source = self.tokenizer.batch_encode_plus(
            source,
            max_length=self.source_maxlength,
            padding='max_length',
            return_tensors='pt',
            truncation=True if self.source_maxlength > 0 else False,
        )
        source_ids = source["input_ids"]
        source_mask = source["attention_mask"].bool()
        # encode candidate text
        candidate_texts = [example['candidates'] for example in batch]
        candidate_ids, candidate_masks = encode_batch_text(candidate_texts,
                                                     self.tokenizer,
                                                     self.source_maxlength)
        # context consists of source and candidate texts for each batch
        # source text being at index 0
        context_ids = torch.cat([source_ids[:, None], candidate_ids], dim=1)
        context_masks = torch.cat([source_mask[:, None], candidate_masks], dim=1)

        scores = torch.stack([example['scores'] for example in batch], dim=0)

        return {
            "input_ids": context_ids,
            "attention_mask": context_masks,
            "labels": target_ids,
            "scores": scores,
        }

class FiDCollator(object):
    def __init__(self, source_maxlength, tokenizer, candidate_maxlength):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        # encode target
        target = self.tokenizer.batch_encode_plus(
            target,
            padding="longest",
            return_tensors='pt',
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)
        # encode FiD texts
        def append_candidates(example):
            return [example['source'] + " " + candidate for candidate in example['candidates']]
        texts = [append_candidates(example) for example in batch]
        context_ids, context_masks = encode_batch_text(texts,
            self.tokenizer,
            self.source_maxlength + self.candidate_maxlength)

        scores = torch.stack([example['scores'] for example in batch], dim=0)

        return {
            "input_ids": context_ids,
            "attention_mask": context_masks,
            "labels": target_ids,
            "scores": scores,
        }

