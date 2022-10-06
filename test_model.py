# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm

import src.util
from src.options import Options
import src.data
import src.model

def evaluate(model, dataset, dataloader, tokenizer, opt):
    rouge_score = ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True).to(torch.device("cuda", 0)) # select score here
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()
    total = 0
    exactmatch = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=200,
                min_length=20,
                num_beams=4,
            )

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    rouge_score.update(ans, example['answers'])

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                result = rouge_score.compute()
                for k, v in result.items():
                    log += f' |\n {k} = {v:.3f}'
                logger.warning(log)
    log = f'Process rank:{opt.global_rank}, final result'
    result = rouge_score.compute()
    for k, v in result.items():
        log += f' |\n {k} = {v:.3f}'
    logger.warning(log)
    if opt.is_distributed:
        torch.distributed.barrier()

    return result


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    model_name = opt.model_type + '-' + opt.model_size
    if opt.model_type == 't5':
        model_name = "t5-" + opt.model_size
        model_class = src.model.FiDT5
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    elif opt.model_type == 'dualt5':
        model_name = "t5-" + opt.model_size
        model_class = src.model.DualFiDT5
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.data.DualCollator(opt.text_maxlength, tokenizer)
    elif opt.model_type == 'bart':
        model_name = "facebook/bart-" + opt.model_size
        model_class = src.model.FiDBART
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    elif opt.model_type == 'dualbart':
        model_name = "facebook/bart-" + opt.model_size
        model_class = src.model.DualFiDBART
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.data.DualCollator(opt.text_maxlength, tokenizer)

    else:
        raise NotImplementedError

    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples,
        opt.n_context,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=20,
        collate_fn=collator_function
    )

    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path)
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)

