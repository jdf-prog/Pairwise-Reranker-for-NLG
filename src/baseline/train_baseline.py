import sys
import os
import torch
import argparse
import logging
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (
    load_raw_dataset,
)
from common.utils import (
    seed_everything,
    str2bool
)
from common.evaluation import (
    overall_eval
)
from model_utils import (
    build_model,
    build_tokenizer,
)

def main(args):
    seed_everything(args.seed)

    tokenizer = build_tokenizer(args)
    model = build_model(args)
    if args.load_checkpoint is not None:
        state_dict = torch.load(os.path.join(args.load_checkpoint, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)
        print("Successfully loaded checkpoint from {}".format(args.load_checkpoint))

    if args.do_train:
        if '1_half' in args.model_name:
            ids, sources, targets = load_raw_dataset(args.dataset, 'train', '1_half')
        elif '2_half' in args.model_name:
            ids, sources, targets = load_raw_dataset(args.dataset, 'train', '2_half')
        else:
            ids, sources, targets = load_raw_dataset(args.dataset, 'train', 'full')
        train_dataset = Dataset(sources, targets, max_size=args.max_train_data_size)
    else:
        train_dataset = None
    if args.do_eval:
        if isinstance(args.eval_steps, int) and args.eval_steps > 0:
            args.evaluation_strategy = 'steps'
            args.save_strategy = 'steps'
            args.save_steps = args.eval_steps
        else:
            args.evaluation_strategy = 'epoch'
            args.save_strategy = 'epoch'
        ids, sources, targets = load_raw_dataset(args.dataset, 'val', 'full')
        eval_dataset = Dataset(sources, targets, max_size=args.max_eval_data_size)
    else:
        eval_dataset = None
        args.evaluation_strategy = 'no'
        args.save_strategy = 'no'
    if args.do_predict:
        ids, sources, targets = load_raw_dataset(args.dataset, 'test', 'full')
        predict_dataset = Dataset(sources, targets, max_size=args.max_predict_data_size)
    else:
        predict_dataset = None

    collator = Collator(tokenizer, args.source_max_length, args.target_max_length, args.prefix)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate,
        generation_num_beams=args.generation_num_beams,
        generation_max_length=args.generation_max_length,
        metric_for_best_model=args.metric_for_best_model,
        logging_steps=args.logging_steps,
        log_level=args.log_level,
        report_to=args.report_to,
        run_name=args.run_name,
        label_smoothing_factor=args.label_smoothing_factor,
        seed=args.seed,
        local_rank=args.local_rank,
        fp16=args.fp16,
        deepspeed=args.deepspeed, #
        sharded_ddp=args.sharded_ddp,
        optim=args.optim,
        remove_unused_columns=False,
        disable_tqdm=False,
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    logging.info(args)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer, args.metrics),
        data_collator=collator,
    )


    if args.do_train:
        if args.evaluate_before_training:
            logging.info('Evaluation before training')
            metrics = trainer.evaluate()
            logging.info(metrics)
        logging.info('Start training...')
        outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logging.info(outputs)
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
        trainer.save_model(best_checkpoint)
        logging.info(f"Best checkpoint saved to {best_checkpoint}")

    if args.do_predict:
        logging.info('Predicting...')
        outputs = trainer.predict(predict_dataset)
        predictions = outputs.predictions
        labels = outputs.label_ids
        metrics = outputs.metrics
        logging.info(f"metrics: {metrics}")
        if args.save_predictions:
            logging.info('Saving predictions...')
            predictions_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            with open(os.path.join(args.output_dir, 'predictions.txt'), 'w') as f:
                f.write("\n".join(predictions_text))
            with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
                f.write("\n".join(labels_text))
            logging.info('Predictions saved to {}'.format(args.output_dir))

from transformers.trainer import (
    unwrap_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)
class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        copy from transformers.trainer.Trainer.compute_loss
        """
        labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

class Collator(object):
    def __init__(self, tokenizer, source_max_length, target_max_length, prefix=None) -> None:
        self.tokenizer = tokenizer
        self.source_max_length = min(source_max_length, tokenizer.model_max_length)
        self.target_max_length = min(target_max_length, tokenizer.model_max_length)
        self.prefix = prefix
        if self.prefix is None:
            self.prefix = ''

    def __call__(self, batch):
        source = [self.prefix + b['source'] for b in batch]
        target = [b['target'] for b in batch]
        source = self.tokenizer(source, padding='longest', max_length=self.source_max_length, truncation=True, return_tensors='pt')
        target = self.tokenizer(target, padding='longest', max_length=self.target_max_length, truncation=True, return_tensors='pt')

        src_len = source['input_ids'].shape[1]
        tgt_len = target['input_ids'].shape[1]
        assert src_len <= self.source_max_length and tgt_len <= self.target_max_length, \
            f"Source length: {src_len}, target length: {tgt_len}"

        source_ids = source['input_ids']
        source_mask = source['attention_mask']
        target_ids = target['input_ids']
        target_mask = target['attention_mask']

        return {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'labels': target_ids,
        }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sources, targets,  max_size=-1):
        self.sources = sources
        self.targets = targets

        if isinstance(max_size, int) and max_size > 0:
            self.sources = self.sources[:max_size]
            self.targets = self.targets[:max_size]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]
        return {
            'source': source,
            'target': target,
        }

def compute_metrics(tokenizer, metrics, eval_pred):
    generation_tokens, labels = eval_pred
    generated = tokenizer.batch_decode(generation_tokens, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    metrics = overall_eval(generated, labels, metrics)
    metrics = {metric: np.mean(value) for metric, value in metrics.items()}
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = "pegasus",
        choices=["pegasus", "bart", "t5", "mt5", "nllb", "opusmt"])
    parser.add_argument('--model', type=str, default = "google/pegasus-large")
    parser.add_argument('--model_name', type=str, default = "pegasus_cnndm_1_half",
        choices = [
            "pegasus_cnndm_1_half", "pegasus_cnndm_2_half", "pegasus_cnndm",
            "bart_cnndm_1_half", "bart_cnndm_2_half", "bart_cnndm",
            "pegasus_xsum_1_half", "pegasus_xsum_2_half", "pegasus_xsum",
            "bart_xsum_1_half", "bart_xsum_2_half", "bart_xsum",
            "t5_wmt18_1_half", "t5_wmt18_2_half", "t5_wmt18",
            "mt5_wmt18_1_half", "mt5_wmt18_2_half", "mt5_wmt18",
            "t5_common_gen_1_half", "t5_common_gen_2_half", "t5_common_gen"
    ])
    parser.add_argument('--dataset', type=str, default = "cnndm",
        choices= ["cnndm", "xsum", "reddit", 'wmt18', 'commongen'])

    # data config
    parser.add_argument('--source_max_length', type=int, default=512)
    parser.add_argument('--target_max_length', type=int, default=128)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--max_train_data_size', type=int, default=-1)
    parser.add_argument('--max_eval_data_size', type=int, default=-1)
    parser.add_argument('--max_predict_data_size', type=int, default=-1)

    # running config
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--fp16', type=str2bool, default=False)
    parser.add_argument('--deepspeed', type=str, default=None) # "ds_config.json"
    parser.add_argument('--sharded_ddp', type=str, default="simple")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # mode
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--do_eval", type=str2bool, default=True)
    parser.add_argument("--do_predict", type=str2bool, default=True)

    # training hyperparameters
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=10e10)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.00)
    parser.add_argument("--warmup_steps", type=int, default=0) # Overrides any effect of :obj:`warmup_ratio`.
    parser.add_argument("--lr_scheduler_type", type=str, choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ], default="linear")
    parser.add_argument('--optim', type=str, default="adafactor")
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--label_smoothing_factor', type=float, default=0.1)

    # logging
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--log_level", type=str, default="passive",
        choices=["passive", "info", "debug", "warning", "error", "critical"])
    parser.add_argument("--report_to", type=str, default='none')
    parser.add_argument("--run_name", type=str, default="basic") # wandb run name

    # evaluation hyperparameters
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--evaluation_strategy", type=str, choices=[
        "steps", "epoch", "no"
    ], default="epoch")
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--evaluate_before_training", type=str2bool, default=False)

    # save config
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=False)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_strategy", type=str, choices=[
        "steps", "epoch", "no"
    ], default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default=None,
        choices=['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu', 'cider']
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)

    # generation config
    parser.add_argument("--predict_with_generate", type=str2bool, default=True)
    parser.add_argument("--generation_num_beams", type=int, default=None)
    parser.add_argument("--generation_max_length", type=int, default=None)


    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"../../models/{args.model_name}"


    # default config
    dataset_names = ["cnndm", "xsum", "reddit", 'wmt18', 'commongen']
    prefix = [None, None, None, "Translate Chinese to English: ", "Generate a sentence with the following words: "]
    metrics = ["rouge1+rouge2+rougeLsum", "rouge1+rouge2+rougeLsum", "rouge1+rouge2+rougeLsum", "bleu", "cider+bleu"]
    # setting default values
    idx = dataset_names.index(args.dataset)
    args.prefix = prefix[idx] if args.prefix is None and 't5' in args.model_type else args.prefix
    args.generation_max_length = args.target_max_length if args.generation_max_length is None else args.generation_max_length
    args.metrics = metrics[idx] if args.metrics is None else args.metrics
    args.metrics = args.metrics.split('+')
    args.metric_for_best_model = args.metrics[0] if args.metric_for_best_model is None else args.metric_for_best_model
    assert args.metric_for_best_model in args.metrics, f"{args.metric_for_best_model} not in {args.metrics}"
    args.cache_dir = "../../hf_models/" + args.model + "/"

    if args.log_level == "passive":
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=args.log_level.upper())

    logging.info(args)
    main(args)

