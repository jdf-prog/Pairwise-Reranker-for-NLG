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
from common.data import (
    save_raw_dataset,
)

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)

# data
parser.add_argument('--dataset', type=str, default = "reddit",
                    choices= ["cnndm", "xsum", "reddit"])

args = parser.parse_args()

dataset_keys = ["cnndm", "xsum", "reddit", 'wmt']
dataset_names = ["cnn_dailymail", "xsum", "reddit_tifu", "wmt18"]
make_splits = [False, False, True, False]
data_versions = ["3.0.0", None, "long", "zh-en"]
source_keys = ["article", "document", "documents", "zh"]
target_keys = ["highlights", "summary", "tldr", "en"]

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

        sources = [x[args.source_key] for x in dataset]
        target = [x[args.target_key] for x in dataset]

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
            save_raw_dataset(args.dataset, set_name, set_sources[idx], set_target[idx])
            idx += 1
    else:
        for x in sets:
            (set, set_name) = x
            dataset_set = dataset[set]
            sources = [x[args.source_key] for x in dataset_set]
            target = [x[args.target_key] for x in dataset_set]
            save_raw_dataset(args.dataset, set_name, sources, target)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



if __name__ == '__main__':

    main(args)
