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
        self.n_candidate = n_candidate if n_candidate is not None and n_candidate > 0 else None
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

def load_data(data_path, args):
    assert data_path, "data_path is not specified"
    print("Loading data from {}".format(data_path))
    if data_path.endswith('.jsonl'):
        with open(data_path) as f:
            data = [json.loads(line) for line in f.readlines()]
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []

    for item in data:
        # debug, only use rouge1, rouge2, rougeL
        for candidate in item['candidates']:
            candidate['scores'] = {
                "rouge1": candidate['scores']['rouge1'],
                "rouge2": candidate['scores']['rouge2'],
                "rougeL": candidate['scores']['rougeLsum'],
            }
        if args.candidate_models is not None:
            item['candidates'] = [candidate for candidate in item['candidates'] if candidate['model'] in args.candidate_models]
        if args.candidate_generation_methods is not None:
            item['candidates'] = [candidate for candidate in item['candidates'] if candidate['generation_method'] in args.candidate_generation_methods]

    for k, example in enumerate(data):
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
        for candidate in example['candidates']:
            candidate['scores'] = {k:float(v) for k,v in list(candidate['scores'].items())}
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


