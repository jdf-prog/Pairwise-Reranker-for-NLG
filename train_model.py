# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
from torchmetrics.text.rouge import ROUGEScore
import wandb

import src.util
import src.data
import src.model

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_rouge, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    torch.manual_seed(3407) # debug
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()


    if opt.is_main:
        wandb.init(project="FID", group=opt.model_type + '-' + opt.model_size, name=opt.name)
        wandb.config.update(opt)

    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch
            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if opt.is_main:
                wandb.log({"train_loss": train_loss.item()})

            if step % opt.eval_freq == 0:
                dev_rouge = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_rouge > best_dev_rouge:
                        best_dev_rouge = dev_rouge
                        src.util.save(model, optimizer, scheduler, step, best_dev_rouge,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_rouge:.2f}ROUGE2 |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_rouge, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.
                    wandb.log({"lr": scheduler.get_last_lr()[0]})
                    wandb.log({"epoch": epoch})



            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_rouge,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    rouge_score = ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    rouge_scores = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=200,
                num_beams=4,
                min_length=20
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                # gold = dataset.get_example(idx[k])['answers']
                # score = src.evaluation.ems(ans, gold)
                gold = dataset.get_example(idx[k])['target']
                score = rouge_score(ans, gold)
                rouge_scores.append(score)
                total += 1

    rouge1_score = np.mean([r['rouge1_fmeasure'] for r in rouge_scores])
    rouge2_score = np.mean([r['rouge2_fmeasure'] for r in rouge_scores])
    rougeL_score = np.mean([r['rougeL_fmeasure'] for r in rouge_scores])
    if opt.is_main:
        wandb.log({"rouge1": rouge1_score})
        wandb.log({"rouge2": rouge2_score})
        wandb.log({"rougeL": rougeL_score})
    return rouge2_score

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = opt.model_type + '-' + opt.model_size
    if opt.model_type == 't5':
        model_name = "t5-" + opt.model_size
        model_class = src.model.FiDT5
        hf_model_class = transformers.T5ForConditionalGeneration
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
        collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    elif opt.model_type == "dualt5":
        model_name = "t5-" + opt.model_size
        model_class = src.model.DualFiDT5
        hf_model_class = transformers.T5ForConditionalGeneration
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
        collator = src.data.DualCollator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    elif opt.model_type == 'bart':
        model_name = "facebook/bart-" + opt.model_size
        model_class = src.model.FiDBART
        hf_model_class = transformers.BartForConditionalGeneration
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name)
        collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    elif opt.model_type == "dualbart":
        model_name = "facebook/bart-" + opt.model_size
        model_class = src.model.DualFiDBART
        hf_model_class = transformers.BartForConditionalGeneration
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name)
        collator = src.data.DualCollator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    else:
        raise NotImplementedError



    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if not checkpoint_exists and opt.model_path == "none":
        hf_model = hf_model_class.from_pretrained(model_name)
        hf_model_class.save_pretrained
        model = model_class(hf_model.config)
        model.load_hfm(hf_model.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_rouge = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_rouge = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_rouge = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")


    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_rouge,
        checkpoint_path
    )
