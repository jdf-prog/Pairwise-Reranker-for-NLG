"""
    This file is modified based on This file is modified based on:
    https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/candidate_generation/main_candidate_generation.py
    We thank the authors for sharing their code.
"""
# Generate summary candidates with the fine-tuned models.

import argparse
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (
    get_candidates
)
from model_utils import (
    build_model,
    build_tokenizer,
    FTModel,
)
from common.utils import (
    seed_everything,
    str2bool
)
from common.data import (
    load_raw_dataset,
    save_pkl_candidates,
    save_pkl_sources_and_targets,
)
from common.dataset import CustomDataset
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = str2bool, default = True)
parser.add_argument('--debug', type = str2bool, default = False)
parser.add_argument('--debug_size', type = int, default = 10)

# data
parser.add_argument('--dataset', type=str, default = "reddit",
                    choices= ["cnndm", "xsum", "reddit"])

# model
parser.add_argument('--model_type', type = str, default = "pegasus",
                    choices=["pegasus", "bart"])
parser.add_argument('--model', type = str, default = "google/pegasus-large",
                    choices = ["google/pegasus-large", "google/pegasus-cnn_dailymail", "google/pegasus-xsum",
                    "facebook/bart-large", "facebook/bart-large-cnn", "facebook/bart-large-xsum",
                    "Helsinki-NLP/opus-mt-zh-en", "Helsinki-NLP/opus-mt-de-en", "Helsinki-NLP/opus-mt-it-en",
                    "facebook/nllb-200-3.3B", "facebook/nllb-200-1.3B", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-distilled-600M",
                    "facebook/m2m100_1.2B", "facebook/m2m100_418M"])
parser.add_argument('--model_name', type=str, default = "pegasus_reddit_train_1",
                    choices = ["pegasus_unsupervised", "bart_unsupervised",
                    "pegasus_cnndm_first_half_shuffled_1", "pegasus_cnndm_second_half_shuffled_1", "pegasus_cnndm",
                    "bart_cnndm_first_half_shuffled_1", "bart_cnndm_second_half_shuffled_1", "bart_cnndm",
                    "pegasus_xsum_first_half_shuffled_1", "pegasus_xsum_second_half_shuffled_1", "pegasus_xsum",
                    "bart_xsum_first_half_shuffled_1", "bart_xsum_second_half_shuffled_1", "bart_xsum",
                    "pegasus_reddit_first_half_shuffled_1", "pegasus_reddit_second_half_shuffled_1", "pegasus_reddit_train_1",
                    "bart_reddit_first_half_shuffled_1", "bart_reddit_second_half_shuffled_1", "bart_reddit_train_1",
                    "opus_mt", "nllb", "m2m100"])
parser.add_argument('--load_model', type = str2bool, default = False)
parser.add_argument('--load_model_path', type = str,
                    default = "../base_model_finetuning/ft_saved_models/reddit/pegasus_reddit_train_1/checkpoint-5/pytorch_model.bin") # todo: change to where you saved the finetuned checkpoint

# summary generation
parser.add_argument('--set', type=str, default = "val",
                    choices = ["train", "first_half_train_shuffled", "second_half_train_shuffled", "val", "test"])
parser.add_argument('--max_val_size', type = int, default = 100000)
parser.add_argument('--inference_bs', type = int, default = 2)
parser.add_argument('--save_candidates', type = str2bool, default = True)
parser.add_argument('--generation_method', type = str, default = "diverse_beam_search",
                    choices = ["beam_search", "diverse_beam_search", "top_p_sampling", "top_k_sampling"])
parser.add_argument('--num_return_sequences', type = int, default = 15) # default: 15
parser.add_argument('--num_beams', type = int, default = 15) # for beam search
parser.add_argument('--num_beam_groups', type = int, default = 15) # for diverse beam search
parser.add_argument('--diversity_penalty', type = float, default = 1.0) # for diverse beam search
parser.add_argument('--top_p', type = float, default = 0.95) # for top-p sampling
parser.add_argument('--top_k', type = int, default = 50) # for top-k sampling
parser.add_argument('--stemmer', type = str2bool, default = True)

args = parser.parse_args()

dataset_names = ["cnndm", "xsum", "reddit", ]
val_data_sizes = [13368, 11332, 4213]
test_data_sizes = [11490, 11334, 4222]
source_max_lengths = [1024, 512, 512] # debug
candidate_max_lengths = [128, 64, 128]
clean_ns = [True, False, False]
length_penalties_pegasus = [0.8, 0.8, 0.6]
length_penalties_bart = [0.8, 0.8, 1.0]
repetition_penalties = [1.0, 1.0, 1.0]
no_repeat_ngram_sizes = [0, 3, 3]

idx = dataset_names.index(args.dataset)

args.source_max_length = source_max_lengths[idx]
args.candidate_max_length = candidate_max_lengths[idx]
args.clean_n = clean_ns[idx]
if args.model_type == "pegasus":
    args.length_penalty = length_penalties_pegasus[idx]
elif args.model_type == "bart":
    args.length_penalty = length_penalties_bart[idx]
args.repetition_penalty = repetition_penalties[idx]
args.no_repeat_ngram_size = no_repeat_ngram_sizes[idx]
args.cache_dir = "../../hf_models/" + args.model.split('/')[-1] + "/"

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)
    data_path = Path("../../data/")
    data_path = data_path / args.dataset / args.set / args.generation_method
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("\nUsing device {}".format(device))

    # data
    data = load_raw_dataset(args.dataset, args.set)

    # tokenizer
    tokenizer = build_tokenizer(args)

    # datasets
    sources, targets = data
    print(len(sources), len(targets))
    sources = sources[:args.max_val_size]
    targets = targets[:args.max_val_size]
    print(len(sources), len(targets))
    if args.debug:
        sources = sources[:args.debug_size]
        targets = targets[:args.debug_size]
    dataset = GenerationDataset(tokenizer, sources, targets, args.source_max_length, args.candidate_max_length)
    print("Total size of dataset: {}".format(len(sources)))

    # data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.inference_bs, shuffle = False)

    # model
    model = build_model(args)
    model = FTModel(model, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_path))
        print("Loaded the model weights!", args.load_model_path)

    # summary generation
    sources, candidates, targets = get_candidates(tokenizer, dataloader, model, device, args)
    # export
    if args.save_candidates:
        save_pkl_sources_and_targets(args.dataset, args.set, sources, targets)
        save_pkl_candidates(args.dataset, args.set, args.generation_method, args.model_name, candidates)

class GenerationDataset(torch.utils.data.Dataset):
    """
        Dataset for generate candidates for given sources
    """

    def __init__(self, tokenizer, sources, targets, source_max_length, target_max_length):
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        source = self.sources[item]
        target = self.targets[item]

        source_inputs = self.tokenizer(source, return_tensors="pt", max_length=self.source_max_length, padding='max_length')
        source_inputs["input_ids"] = source_inputs["input_ids"][:, :self.source_max_length]
        source_inputs["attention_mask"] = source_inputs["attention_mask"][:, :self.source_max_length]

        target_inputs = self.tokenizer(target, return_tensors="pt", max_length=self.target_max_length, padding='max_length')
        target_inputs["input_ids"] = target_inputs["input_ids"][:, :self.target_max_length]
        target_inputs["attention_mask"] = target_inputs["attention_mask"][:, :self.target_max_length]

        batch = {
            "source": source,
            "source_inputs": source_inputs,
            "target": target,
            "target_inputs": target_inputs,
        }

        return batch


if __name__ == '__main__':

    main(args)
