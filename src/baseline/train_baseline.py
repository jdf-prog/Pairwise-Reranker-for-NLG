import sys
import os
import torch
import torch.nn as nn
import argparse
import logging
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (
    load_raw_dataset,
)
from common.utils import (
    seed_everything,
    str2bool
)
from model_utils import (
    build_model,
    build_tokenizer,
)
logger = logging.getLogger(__name__)

def main(args):
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        log_level=args.log_level,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        local_rank=args.local_rank,
        fp16=args.fp16,
        deepspeed=args.deepspeed, #
        sharded_ddp=args.sharded_ddp,
        evaluation_strategy='no',
        save_strategy='no',
        adafactor=args.adafactor,
        remove_unused_columns=False,
        disable_tqdm=False,
        greater_is_better=True,
    )
    tokenizer = build_tokenizer(args)
    model = build_model(args)

    if '1_half' in args.model_name:
        sources, targets = load_raw_dataset(args.dataset, 'train', '1_half')
    elif '2_half' in args.model_name:
        sources, targets = load_raw_dataset(args.dataset, 'train', '2_half')
    else:
        sources, targets = load_raw_dataset(args.dataset, 'train', 'full')

    dataset = Dataset(tokenizer, sources, targets, args.source_max_length, args.target_max_length, args.prefix)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    trainer.save_model(args.output_dir)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, sources, targets, source_max_length, target_max_length, prefix=None):
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.source_max_length = min(source_max_length, tokenizer.model_max_length)
        self.target_max_length = min(target_max_length, tokenizer.model_max_length)
        self.prefix = prefix
        if self.prefix is None:
            self.prefix = ''

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.prefix + self.sources[idx]
        target = self.targets[idx]

        source = self.tokenizer(source, max_length=self.source_max_length, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer(target, max_length=self.target_max_length, padding='max_length', truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze(0)
        source_mask = source['attention_mask'].squeeze(0)
        target_ids = target['input_ids'].squeeze(0)
        target_mask = target['attention_mask'].squeeze(0)

        return {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'labels': target_ids,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = "pegasus",
        choices=["pegasus", "bart", "t5", "mt5", "nllb"])
    parser.add_argument('--model', type=str, default = "google/pegasus-large",
        choices = ["google/pegasus-large", "facebook/bart-large", "t5-large", "google/mt5-large"])
    parser.add_argument('--model_name', type=str, default = "pegasus_cnndm_1_half",
        choices = [
            "pegasus_cnndm_1_half", "pegasus_cnndm_2_half"
            "bart_cnndm_1_half", "bart_cnndm_2_half"
            "pegasus_xsum_1_half", "pegasus_xsum_2_half"
            "bart_xsum_1_half", "bart_xsum_2_half"
            "mt5_wmt18_1_half", "mt5_wmt18_2_half"
            "t5_common_gen_1_half", "t5_common_gen_2_half"
    ])
    parser.add_argument('--dataset', type=str, default = "cnndm",
        choices= ["cnndm", "xsum", "reddit", 'wmt18', 'commongen'])

    # data config
    parser.add_argument('--source_max_length', type=int, default=None)
    parser.add_argument('--target_max_length', type=int, default=None)
    parser.add_argument('--prefix', type=str, default=None)

    # running config
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument('--deepspeed', type=str, default=None) # "ds_config.json"
    parser.add_argument('--sharded_ddp', type=str, default="simple")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # mode
    parser.add_argument("--do_train", type=str2bool, default=True)

    # training hyperparameters
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=10e10)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0) # Overrides any effect of :obj:`warmup_ratio`.
    parser.add_argument("--lr_scheduler_type", type=str, choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ], default="linear")
    parser.add_argument('--adafactor', type=bool, default=True)

    # logging
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--log_level", type=str, default="passive",
        choices=["passive", "info", "debug", "warning", "error", "critical"])
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="basic") # wandb run name

    # save config
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=False)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"../../models/{args.model_name}"

    dataset_names = ["cnndm", "xsum", "reddit", 'wmt18', 'commongen']
    source_max_lengths = [1024, 512, 512, 512, 10] # debug
    target_max_lengths = [128, 64, 128, 350, 35]
    lr = [5e-5, 5e-5, 1e-4, 5e-5, 5e-5]
    prefix = [
        None,
        None,
        None,
        "Translate Chinese to English: ",
        "Generate a sentence with the following words: "
    ]
    per_device_train_batch_size = [4, 4, 4, 4, 128]
    gradient_accumulation_steps = [64, 64, 64, 64, 2]
    num_train_epochs = [10, 10, 15, 10, 10]


    # default config
    idx = dataset_names.index(args.dataset)
    args.source_max_length = source_max_lengths[idx]
    args.target_max_length = target_max_lengths[idx]
    args.learning_rate = lr[idx] if args.learning_rate is None else args.learning_rate
    args.prefix = prefix[idx] if args.prefix is None and 't5' in args.model_type else args.prefix
    args.per_device_train_batch_size = per_device_train_batch_size[idx] if args.per_device_train_batch_size is None else args.per_device_train_batch_size
    args.gradient_accumulation_steps = gradient_accumulation_steps[idx] if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
    args.num_train_epochs = num_train_epochs[idx] if args.num_train_epochs is None else args.num_train_epochs

    if args.log_level == "passive":
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=args.log_level.upper())
    logger = logging.getLogger(__name__)
    logging.info(args)
    main(args)

