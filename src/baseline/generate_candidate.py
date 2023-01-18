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
    str2bool,
    empty2None,
)
from common.data import (
    load_raw_dataset,
    save_pkl_candidates,
    save_pkl_sources_and_targets,
    exist_pkl_candidates
)
from common.dataset import CustomDataset
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = str2bool, default = True)
parser.add_argument('--debug', type = str2bool, default = False)
parser.add_argument('--debug_size', type = int, default = 10)

# data
parser.add_argument('--dataset', type=str, default = "cnndm",
                    choices= ["cnndm", "xsum", "reddit", 'wmt18', 'commongen'])

# model
parser.add_argument('--model_type', type = str, default = "pegasus",
                    choices=["pegasus", "bart", "opus-mt", "t5", "flan-t5", "nllb"])
parser.add_argument('--model', type = str, default = "google/pegasus-large",
                    choices = ["google/pegasus-large", "google/pegasus-cnn_dailymail", "google/pegasus-xsum",
                    "facebook/bart-large", "facebook/bart-large-cnn", "facebook/bart-large-xsum",
                    "Helsinki-NLP/opus-mt-zh-en", "Helsinki-NLP/opus-mt-de-en", "Helsinki-NLP/opus-mt-it-en",
                    "facebook/nllb-200-3.3B", "facebook/nllb-200-1.3B", "facebook/nllb-200-distilled-1.3B", "facebook/nllb-200-distilled-600M",
                    "facebook/m2m100_1.2B", "facebook/m2m100_418M", "t5-large", "t5-base",
                    "mrm8488/t5-base-finetuned-common_gen", "google/flan-t5-large", "sibyl/BART-large-commongen"])
parser.add_argument('--model_name', type=str, default = "pegasus_reddit_train_1",
                    choices = [
                        "pegasus_cnndm", "pegasus_cnndm_1_half", "pegasus_cnndm_2_half", "pegasus_cnndm_half",
                        "bart_cnndm", "bart_cnndm_1_half", "bart_cnndm_2_half", "bart_cnndm_half",
                        "pegasus_xsum", "pegasus_xsum_1_half", "pegasus_xsum_2_half", "pegasus_xsum_half",
                        "bart_xsum", "bart_xsum_1_half", "bart_xsum_2_half", "bart_xsum_half",
                        "t5_wmt18_1_half", "t5_wmt18_2_half", "t5_wmt18_half",
                        "t5_common_gen_1_half", "t5_common_gen_2_half", "t5_common_gen_half",
                        "opus_mt", "nllb-3.3B", "nllb-1.3B", "nllb-600M", "m2m100",
                        'flan-t5-large', 'flan-t5-base', 't5_common_gen', 'bart_common_gen',
                        't5_common_gen_1_beam'])
parser.add_argument('--load_model', type = str2bool, default = False)
parser.add_argument('--load_model_path', type = str, default = None)

# summary generation
parser.add_argument('--set', type=str, default = "val",
                    choices = ["train", "first_half_train_shuffled", "second_half_train_shuffled", "val", "test"])
parser.add_argument('--max_val_size', type = int, default = -1)
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

# generation config
parser.add_argument('--source_max_length', type = int, default = None)
parser.add_argument('--candidate_max_length', type = int, default = None)
parser.add_argument('--length_penalty', type = float, default = None)
parser.add_argument('--repetition_penalty', type = float, default = None)
parser.add_argument('--no_repeat_ngram_size', type = int, default = None)


parser.add_argument('--start_idx', type = int, default = None)
parser.add_argument('--end_idx', type = int, default = None)
parser.add_argument('--partition', type = empty2None, default = None,
    choices = ['1_half', '2_half', 'full', None])
parser.add_argument('--overwrite', type = str2bool, default = True)
args = parser.parse_args()

dataset_names = ["cnndm", "xsum", "reddit", 'wmt18', 'commongen']
source_max_lengths = [1024, 512, 512, 512, 35]
candidate_max_lengths = [128, 64, 128, 350, 35]
clean_ns = [True, False, False, False, False]
length_penalties_pegasus = [0.8, 0.8, 0.6, 0.8, 0.8]
length_penalties_bart = [0.8, 0.8, 1.0, 0.8, 0.8]
repetition_penalties = [1.0, 1.0, 1.0, 1.0, 1.0]
no_repeat_ngram_sizes = [0, 3, 3, 0, 0]
prefix = [None, None, None, "Translate Chinese to English: ", "Generate a sentence with the following words: "]

idx = dataset_names.index(args.dataset)

args.source_max_length = source_max_lengths[idx] if args.source_max_length is None else args.source_max_length
args.candidate_max_length = candidate_max_lengths[idx] if args.candidate_max_length is None else args.candidate_max_length
args.clean_n = clean_ns[idx]
if args.length_penalty is None:
    if args.model_type == "pegasus":
        args.length_penalty = length_penalties_pegasus[idx]
    elif args.model_type == "bart":
        args.length_penalty = length_penalties_bart[idx]
    else:
        args.length_penalty = 1.0
args.repetition_penalty = repetition_penalties[idx] if args.repetition_penalty is None else args.repetition_penalty
args.no_repeat_ngram_size = no_repeat_ngram_sizes[idx] if args.no_repeat_ngram_size is None else args.no_repeat_ngram_size
args.cache_dir = "../../hf_models/" + args.model + "/"
args.prefix = prefix[idx]

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

    # tokenizer
    tokenizer = build_tokenizer(args)
    if "nllb" in args.model:
        forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
    else:
        forced_bos_token_id = None

    # data
    ids, sources, targets, offsets = load_raw_dataset(args.dataset, args.set, partition=args.partition, return_offsets=True)

    if args.start_idx is not None and args.end_idx is not None:
        print("Using start_idx: {}, end_idx: {}".format(args.start_idx, args.end_idx))
        ids = ids[args.start_idx:args.end_idx]
        sources = sources[args.start_idx:args.end_idx]
        targets = targets[args.start_idx:args.end_idx]
        if len(sources) < args.end_idx - args.start_idx:
            print("End index is larger than the dataset size. Using the last index instead.")
            args.end_idx = len(sources) + args.start_idx
        print("Current data size: {}".format(len(sources)))
        # add offsets for saving
        args.start_idx += offsets[0]
        args.end_idx += offsets[0]
    elif args.partition in ['1_half', '2_half']:
        print("Using start_idx: {}, end_idx: {}".format(offsets[0], offsets[1]))
        args.start_idx = offsets[0]
        args.end_idx = offsets[1]

    print("Idxs used for saving: {} - {}".format(args.start_idx, args.end_idx))

    if isinstance(args.max_val_size, int) and args.max_val_size > 0:
        print("Cutting data to {} below samples".format(args.max_val_size))
        ids = ids[:args.max_val_size]
        sources = sources[:args.max_val_size]
        targets = targets[:args.max_val_size]
        print("Current data size: {}".format(len(sources)))
    if args.debug:
        print(f"Debug mode: cutting data to {args.debug_size} samples")
        ids = ids[:args.debug_size]
        sources = sources[:args.debug_size]
        targets = targets[:args.debug_size]
        print("Current data size: {}".format(len(sources)))

    if len(sources) == 0:
        print("No data to evaluate")
        return

    # check if the data have already been generated
    if exist_pkl_candidates(args.dataset, args.set, args.generation_method, args.model_name, start_idx=args.start_idx, end_idx=args.end_idx):
        print("Found existing candidates.")
        if args.overwrite:
            print("Overwriting existing data")
        else:
            print("Not overwriting existing data. Finishing generating")
            return

    dataset = GenerationDataset(tokenizer, sources, targets, args.source_max_length, args.candidate_max_length, args.prefix)
    print("Total size of dataset: {}".format(len(sources)))
    # data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.inference_bs, shuffle = False)
    # model
    model = build_model(args)
    if args.load_model:
        state_dict = torch.load(os.path.join(args.load_model_path, "pytorch_model.bin"))
        load_result = model.load_state_dict(state_dict)
        print("Loaded the model weights!", args.load_model_path)
    model = FTModel(model, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)
    # summary generation
    candidates = get_candidates(tokenizer, dataloader, model, device, args, forced_bos_token_id=forced_bos_token_id)

    # export
    if args.save_candidates:
        save_pkl_sources_and_targets(args.dataset, args.set, sources, targets, start_idx=args.start_idx, end_idx=args.end_idx)
        save_pkl_candidates(args.dataset, args.set, args.generation_method, args.model_name, candidates, start_idx=args.start_idx, end_idx=args.end_idx)

class GenerationDataset(torch.utils.data.Dataset):
    """
        Dataset for generate candidates for given sources
    """

    def __init__(self, tokenizer, sources, targets, source_max_length, target_max_length, prefix=None):
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.source_max_length = min(source_max_length, tokenizer.model_max_length)
        self.target_max_length = min(target_max_length, tokenizer.model_max_length)
        self.prefix = prefix if prefix is not None else ""

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.prefix + self.sources[idx]
        source_inputs = self.tokenizer(source, max_length=self.source_max_length, padding='max_length', truncation=True, return_tensors="pt")
        batch = {
            "source": source,
            "source_inputs": source_inputs,
        }

        return batch


if __name__ == '__main__':
    main(args)
