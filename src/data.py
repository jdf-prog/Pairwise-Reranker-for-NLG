# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_candidate=None,
                 source_prefix='document: ',
                 candidate_prefix='summary: '):
        self.data = data
        self.n_candidate = n_candidate
        self.source_prefix = source_prefix
        self.candidate_prefix = candidate_prefix

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        return {
            'index' : index,
            'source' : self.source_prefix + self.data[index]['source'],
            'target' : self.get_target(self.data[index]),
            'candidates' : [(self.candidate_prefix + "{}").format(c['text']) for c in self.data[index]['candidates'][:self.n_candidate]]  if ('candidates' in self.data[index] and self.n_candidate is not None) else None,
            'scores' : torch.tensor([[float(score) for score in c['scores']] for c in self.data[index]['candidates'][:self.n_candidate]]) if ('candidates' in self.data[index] and self.n_candidate is not None) else None,
        }

    def get_example(self, index):
        return self.data[index]

def encode_candidates(batch_text_candidates, tokenizer, max_length):
    candidate_ids, candidate_masks = [], []
    for k, text_candidates in enumerate(batch_text_candidates):
        p = tokenizer.batch_encode_plus(
            text_candidates,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        candidate_ids.append(p['input_ids'][None])
        candidate_masks.append(p['attention_mask'][None])

    candidate_ids = torch.cat(candidate_ids, dim=0)
    candidate_masks = torch.cat(candidate_masks, dim=0)
    return candidate_ids, candidate_masks.bool()

class DualCollator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)
        context_texts = [[example['source']] + example['candidates'] for example in batch]
        context_ids, context_masks = encode_candidates(context_texts,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        scores = torch.stack([
            [
                list(candidate['scores'].values())
                for candidate in example['candidates']
            ]
            for example in batch
        ], dim=0)

        return (index, target_ids, target_mask, context_ids, context_masks, scores)

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_source(example):
            if example['candidates'] is None:
                return [example['source']]
            return [t + " " + example['source'] for t in example['candidates']]
        text_candidates = [append_source(example) for example in batch]
        candidate_ids, candidate_masks = encode_candidates(text_candidates,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, candidate_ids, candidate_masks)

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples
