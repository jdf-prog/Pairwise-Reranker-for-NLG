# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
from common.evaluation import METRIC_WEIGHTS

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_candidates=None):
        self.data = data
        self.n_candidates = n_candidates if n_candidates is not None and n_candidates > 0 else None
        self.n_tasks = len(self.data[0]['candidates'][0]['scores']) if 'candidates' in self.data[0] else -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.data[index]["target"]
        if isinstance(target, list):
            target = target[0]
        return {
            'index' : index,
            'source' : self.data[index]['source'],
            'target' : target,
            'candidates' : ["{}".format(c['text']) for c in self.data[index]['candidates'][:self.n_candidates]]  if ('candidates' in self.data[index] and self.n_candidates is not None) else None,
            'scores' : [[float(score) for score in c['scores'].values()] for c in self.data[index]['candidates'][:self.n_candidates]] if ('candidates' in self.data[index] and self.n_candidates is not None) else None,
        }

    def get_example(self, index):
        return self.data[index]


def load_data(data_path, args, max_size=None, reset_scores=False):
    assert data_path, "data_path is not specified"
    print("Loading data from {}".format(data_path))
    if data_path.endswith('.jsonl'):
        with open(data_path) as f:
            data = [json.loads(line) for line in f.readlines()]
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    else:
        raise ValueError("Unknown data")
    if max_size is not None and max_size > 0:
        data = data[:max_size]
    examples = []

    for item in data:
        if args.candidate_models is not None:
            item['candidates'] = [candidate for candidate in item['candidates'] if candidate['model'] in args.candidate_models]
        if args.candidate_generation_methods is not None:
            item['candidates'] = [candidate for candidate in item['candidates'] if candidate['generation_method'] in args.candidate_generation_methods]
        if len(item['candidates']) == 0:
            raise ValueError("No candidates left after filtering")

        for candidate in item['candidates']:
            candidate['scores'] = {
                metric: candidate['scores'][metric] for metric in args.metrics
            }

        if reset_scores:
            for generation_method in args.candidate_generation_methods:
                method_candidates = [candidate for candidate in item['candidates'] if candidate['generation_method'] == generation_method]
                sorted_method_candidates = sorted(method_candidates, key=lambda x: np.mean(list(x['scores'].values())), reverse=True)

                # randomly assign the worst candidate for each generation method a negeative score
                for metric in sorted_method_candidates[-1]['scores']:
                    sorted_method_candidates[-1]['scores'][metric] = -random.random()
            sorted_candidates = sorted(item['candidates'], key=lambda x: np.mean(list(x['scores'].values())), reverse=True)
            worst_candidate_methods = [candidate['generation_method'] for candidate in sorted_candidates[-len(args.candidate_generation_methods):]]
            assert set(worst_candidate_methods) == set(args.candidate_generation_methods)

    for k, example in enumerate(data):
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
        for candidate in example['candidates']:
            candidate['scores'] = {k:float(v) for k,v in list(candidate['scores'].items())}
    examples = check_and_normalize_scores(examples)
    return examples

def check_and_normalize_scores(examples):
    """
        Check the upper bound of the scores and print it
    """
    n_candidates = len(examples[0]['candidates'])
    task_names = list(examples[0]['candidates'][0]['scores'].keys())
    max_scores_per_group = {task:[] for task in task_names}
    scores = {task:[] for task in task_names}
    for example in examples:
        for task in task_names:
            scores[task].extend([c['scores'][task] for c in example['candidates']])
            max_scores_per_group[task].append(max([c['scores'][task] for c in example['candidates']]))
    # print checked scores
    for task in task_names:
        print(f"Selection Upper bound for task '{task}' is {np.mean(max_scores_per_group[task])}")
    candidate_scores = {task:[np.mean([ex['candidates'][i]['scores'][task] for ex in examples]) for i in range(n_candidates)] for task in task_names}
    for task in task_names:
        print(f"Candidate mean scores for task '{task}' are {candidate_scores[task]}")

    # normalize scores if training dataset
    metric_weights = METRIC_WEIGHTS

    for example in examples:
        for candidate in example['candidates']:
            for task in task_names:
                candidate['scores'][task] *= metric_weights[task]
    return examples





