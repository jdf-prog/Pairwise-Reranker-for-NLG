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
                 source_prefix='source text: ',
                 candidate_prefix='candidate text: '):
        self.data = data
        self.n_candidate = n_candidate
        self.source_prefix = source_prefix
        self.candidate_prefix = candidate_prefix
        self.n_tasks = len(self.data[0]['candidates'][0]['scores']) if 'candidates' in self.data[0] else -1

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
            'scores' : torch.tensor([[float(score) for score in c['scores'].values()] for c in self.data[index]['candidates'][:self.n_candidate]]) if ('candidates' in self.data[index] and self.n_candidate is not None) else None,
        }

    def get_example(self, index):
        return self.data[index]

def encode_batch_text(batch_texts, tokenizer, max_length):
    encoded_ids, encoded_masks = [], []
    for k, texts in enumerate(batch_texts):
        p = tokenizer.batch_encode_plus(
            texts,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        encoded_ids.append(p['input_ids'][None])
        encoded_masks.append(p['attention_mask'][None])

    encoded_ids = torch.cat(encoded_ids, dim=0)
    encoded_masks = torch.cat(encoded_masks, dim=0)
    return encoded_ids, encoded_masks.bool()

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

        return (index, target_ids, target_mask, context_ids, context_masks, scores)

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
            return [candidate + " " + example['source'] for candidate in example['candidates']]
        texts = [append_candidates(example) for example in batch]
        context_ids, context_masks = encode_batch_text(texts,
            self.tokenizer,
            self.source_maxlength + self.candidate_maxlength)

        scores = torch.stack([example['scores'] for example in batch], dim=0)

        return (index, target_ids, target_mask, context_ids, context_masks, scores)

def load_data(data_path=None, global_rank=-1, world_size=-1, n_tasks=-1):
    assert data_path
    print("Loading data from {}".format(data_path))
    if data_path.endswith('.jsonl'):
        with open(data_path) as f:
            data = [json.loads(line) for line in f.readlines()]
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []

    # debug
    for item in data:
        for candidate in item['candidates']:
            candidate['scores'] = {
                "rouge1": candidate['scores']['rouge1'],
                "rouge2": candidate['scores']['rouge2'],
                "rougeL": candidate['scores']['rougeL'],
            }

    if n_tasks < 0:
        n_tasks = len(data[0]['candidates'][0]['scores'])
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
        for candidate in example['candidates']:
            candidate['scores'] = {k:float(v) for k,v in list(candidate['scores'].items())[:n_tasks]}
            assert len(candidate['scores']) == n_tasks, f"{len(candidate['scores'])} != {n_tasks}"
    check_scores(examples)
    return examples

def check_scores(examples):
    """
        Check the upper bound of the scores and print it
    """
    n_candidate = len(examples[0]['candidates'])
    task_names = list(examples[0]['candidates'][0]['scores'].keys())
    max_scores = {task:[] for task in task_names}
    for example in examples:
        for task in task_names:
            max_scores[task].append(max([c['scores'][task] for c in example['candidates']]))
    candidate_scores = {task:[np.mean([ex['candidates'][i]['scores'][task] for ex in examples]) for i in range(n_candidate)] for task in task_names}
    for task in task_names:
        print(f"Selection Upper bound for task '{task}' is {np.mean(max_scores[task])}")
    for task in task_names:
        print(f"Candidate mean scores for task '{task}' are {candidate_scores[task]}")

