import argparse
import sys
import os
import psutil
from tqdm.contrib.concurrent import process_map
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import (
    load_prepared_dataset,
    save_prepared_dataset
)
from common.dataset import (
    CustomDataset,
)
from common.utils import (
    seed_everything,
    str2bool,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnn_dailymail", choices=["cnn_dailymail", "xsum", "gigaword"])
    parser.add_argument("--set", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--metrics", type=str, nargs="+", default=["rouge1", "rouge2", "rougeL", "rougeLsum", "bleu", "cider", "spice"])
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--generation_methods", type=str, nargs="+", default=None)

    args = parser.parse_args()
    args.metrics = args.metrics.split("+") if args.metrics is not None else None
    args.models = args.models.split("+") if args.models is not None else None
    args.generation_methods = args.generation_methods.split("+") if args.generation_methods is not None else None
    ds = load_prepared_dataset(args.dataset, args.set, args.models, args.generation_methods, args.metrics)

    save_prepared_dataset(args.dataset, args.set, ds)




