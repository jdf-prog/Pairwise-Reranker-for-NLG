"""
    Eval results will be continuously saved to ../../data/prepared/{dataset_name}/{set_name}/dataset.jsonl
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import (
    load_pkl_candidates,
    load_prepared_dataset,
    save_prepared_dataset,
    get_candidate_types
)
from common.utils import (
    seed_everything,
    str2bool,
)
from pathlib import Path

def main(args):
    # seed
    seed_everything(args.seed)
    ds = load_prepared_dataset(args.dataset, args.set)

    # model and generation_method of current computed candidates
    types = get_candidate_types(args.dataset, args.set)
    model_names = set([t[0] for t in types])
    generation_methods = set([t[1] for t in types])

    # add candidates if the curent model and generation_method are not in the dataset
    to_load_types = set(types) - set(ds.candidate_counts.keys())
    for model_name, generation_method in to_load_types:
        print("Loading candidates from -- model:{} \t generation method:{}".format(model_name, generation_method))
        candidates = load_pkl_candidates(args.dataset, args.set, generation_method, model_name)[:1000] # debug
        ds.add_candidates(model_name, generation_method, candidates)

    # prepare metrics
    metrics = []
    if args.eval_rouge:
        metrics.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])
    if args.eval_bleu:
        metrics.extend(["bleu"])
    if args.eval_bleurt:
        metrics.extend(["bleurt"])
    if args.eval_bertscore:
        metrics.extend(["bertscore"])
    ds.prepare_metrics(metrics)
    save_prepared_dataset(args.dataset, args.set, ds)

    # analyze the oracle
    ds.analyze_oracle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--set", type=str, default="val")
    parser.add_argument("--seed", type=int, default=42)
    # metrics
    parser.add_argument('--eval_rouge', type = str2bool, default = True)
    parser.add_argument('--eval_bleu', type = str2bool, default = True)
    parser.add_argument('--eval_bleurt', type = str2bool, default = False)
    parser.add_argument('--eval_bertscore', type = str2bool, default = False)
    args = parser.parse_args()
    main(args)
