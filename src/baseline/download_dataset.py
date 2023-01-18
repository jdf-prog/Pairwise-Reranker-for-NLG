"""
    This script downloads the dataset and splits it into train, val, test
    This file is modified based on:
    https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/candidate_generation/main_download_dataset.py
    We thank the authors for sharing their code.
"""

import time
import os
import numpy as np
import random
import argparse
import sys
import datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List, Tuple, Union
from common.data import save_raw_dataset
from common.utils import seed_everything

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)

# data
parser.add_argument('--dataset', type=str, default = "cnndm",
                    choices= ["cnndm", "xsum", "reddit", 'wmt18', 'commongen'])

parser.add_argument('--max_size', type = int, default = 1000000)
parser.add_argument('--shuffle_train', action='store_true')
parser.add_argument('--agg_target', action='store_true')

args = parser.parse_args()

dataset_keys = ["cnndm", "xsum", "reddit", 'wmt18', 'commongen']
dataset_names = ["cnn_dailymail", "xsum", "reddit_tifu", "wmt18", 'common_gen']
make_splits = [False, False, True, False, False]
data_versions = ["3.0.0", None, "long", "zh-en", None]
source_keys = ["article", "document", "documents", "zh", "concepts"]
target_keys = ["highlights", "summary", "tldr", "en", "target"]

idx = dataset_keys.index(args.dataset)
args.dataset_name = dataset_names[idx]
args.make_split = make_splits[idx]
args.data_version = data_versions[idx]
args.source_key = source_keys[idx]
args.target_key = target_keys[idx]

print("*"*50)
print(args)

sets = [("validation", "val"), ("test", "test"), ("train", "train")]



def main(args):
    seed_everything(args.seed)
    data_path = Path("../../data/raw") / args.dataset
    data_path.mkdir(parents=True, exist_ok=True)

    print(args.dataset_name, args.data_version)
    dataset = datasets.load_dataset(args.dataset_name, args.data_version)

    if args.make_split:
        dataset = dataset["train"]
        if 'wmt' in args.dataset:
            slang, tlang = args.data_version.split('-')
            sources = [d[slang] for d in dataset['translation']]
            targets = [d[tlang] for d in dataset['translation']]
        else:
            sources = [x[args.source_key] for x in dataset]
            target = [x[args.target_key] for x in dataset]
        ids = np.random.permutation(len(sources))[:(args.max_size)]
        sources = [sources[i] for i in ids]
        target = [target[i] for i in ids]
        sources = process_sources(sources)


        idx = np.random.permutation(len(sources))
        sources = [sources[i] for i in idx]
        target = [target[i] for i in idx]

        print(len(sources), len(target))
        print(sources[0])
        print("*" * 50)
        print(target[0])

        thresh = int(0.1 * len(sources))
        train_sources = sources[:(8 * thresh)]
        train_target = target[:(8 * thresh)]
        val_sources = sources[(8 * thresh):(9 * thresh)]
        val_target = target[(8 * thresh):(9 * thresh)]
        test_sources = sources[(9 * thresh):]
        test_target = target[(9 * thresh):]
        print(len(train_sources), len(val_sources), len(test_sources))

        set_sources = [train_sources, val_sources, test_sources]
        set_target = [train_target, val_target, test_target]
        set_names = ["train", "val", "test"]
        idx = 0
        for set_name in set_names:
            save_raw_dataset(args.dataset, set_name, set_sources[idx], set_target[idx], args.shuffle_train and set_name=="train")
            idx += 1
    else:
        for x in sets:
            (set, set_name) = x
            dataset_set = dataset[set]
            if 'wmt' in args.dataset:
                slang, tlang = args.data_version.split('-')
                sources = [d[slang] for d in dataset_set['translation']]
                targets = [d[tlang] for d in dataset_set['translation']]
            elif 'commongen' in args.dataset:
                if set_name == 'train' or (not args.agg_target):
                    sources = [x[args.source_key] for x in dataset_set]
                    sources = process_sources(sources)
                    targets = [x[args.target_key] for x in dataset_set]
                else:
                    items = {}
                    for x in dataset_set:
                        if x['concept_set_idx'] not in items:
                            items[x['concept_set_idx']] = {}
                            items[x['concept_set_idx']]['source'] = x[args.source_key]
                            items[x['concept_set_idx']]['target'] = [x[args.target_key]]
                        else:
                            assert x[args.source_key] == items[x['concept_set_idx']]['source'], f"Source mismatch: {x[args.source_key]} vs {items[x['concept_set_idx']]['source']}"
                            items[x['concept_set_idx']]['target'].append(x[args.target_key])
                    keys = list(items.keys())
                    keys = sorted(keys)
                    sources = [items[key]['source'] for key in keys]
                    sources = process_sources(sources)
                    targets = [items[key]['target'] for key in keys]
            else:
                sources = [x[args.source_key] for x in dataset_set]
                targets = [x[args.target_key] for x in dataset_set]
            save_raw_dataset(args.dataset, set_name, sources, targets, args.shuffle_train and set_name=="train", args.max_size)

def process_sources(sources:List[Union[str, List[str]]]):
    """
        Used for common gen concepts
    """
    new_sources = []
    for source in sources:
        if isinstance(source, list):
            new_sources.append(" ".join(source))
        else:
            new_sources.append(source)
    return new_sources



if __name__ == '__main__':

    main(args)
