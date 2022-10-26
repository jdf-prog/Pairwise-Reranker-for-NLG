import torch
import wandb
import torch.nn as nn
import numpy as np
import os
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import EvalPrediction
from typing import Dict, List, Optional, Tuple, Union, Any
from common.evaluation import (
    eval_rouge,
    eval_bleu,
)
class RerankerTrainer(Trainer):
    def evaluate(
        self,
        **kwargs,
    ) -> Dict[str, float]:
        metrics = super().evaluate(**kwargs)
        if self.is_world_process_zero():
            wandb.log(metrics)
        return metrics

    def save_model(self, output_dir: Optional[str] = None):
        if self.is_world_process_zero():
            super().save_model(output_dir)
            model = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(model.args, os.path.join(output_dir, "config.bin"))


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    pred_scores, scores = eval_pred # pred_scores [batch_size, num_candidates], scores [batch_size, num_candidates, n_tasks]
    agg_scores = np.sum(scores, axis=-1) # aggregate scores

    sort_indices = np.flip(np.argsort(agg_scores, axis=-1), axis=-1) # (batch_size, n_candidate), expected ranks
    ranks = np.zeros_like(sort_indices)
    ranks[np.arange(sort_indices.shape[0])[:, None], sort_indices] = np.arange(sort_indices.shape[-1])
    pred_sort_indices = np.flip(np.argsort(pred_scores, axis=-1), axis=-1) # (batch_size, n_candidate), predicted ranks
    pred_ranks = np.zeros_like(pred_sort_indices)
    pred_ranks[np.arange(pred_sort_indices.shape[0])[:, None], pred_sort_indices] = np.arange(pred_sort_indices.shape[-1])

    # compute selection scores
    sel_idx = np.argmax(pred_scores, axis=1) # [batch_size]
    sel_scores = scores[np.arange(scores.shape[0]), sel_idx] # [batch_size, n_task]
    sel_ranks = ranks[np.arange(ranks.shape[0]), sel_idx] # [batch_size]
    sel_acc = np.mean((sel_ranks == 0)) # scalar

    # compute oracle scores for reference
    oracle_sel_idx = np.argmax(agg_scores, axis=1) # [batch_size]
    oracle_sel_scores = scores[np.arange(scores.shape[0]), oracle_sel_idx] # [batch_size, n_task]
    oracle_sel_ranks = ranks[np.arange(ranks.shape[0]), oracle_sel_idx] # [batch_size]
    oracle_sel_acc = np.mean((oracle_sel_ranks == 0)) # scalar

    metrics = {
        "sel": {
            "acc": sel_acc,
            "rank": np.mean(sel_ranks),
            "rouge1": np.mean(sel_scores[:, 0]),
            "rouge2": np.mean(sel_scores[:, 1]),
            "rougeL": np.mean(sel_scores[:, 2]),
        },
        "oracle": {
            "acc": oracle_sel_acc,
            "rank": np.mean(oracle_sel_ranks),
            "rouge1": np.mean(oracle_sel_scores[:, 0]),
            "rouge2": np.mean(oracle_sel_scores[:, 1]),
            "rougeL": np.mean(oracle_sel_scores[:, 2]),
        },
        "dev_score": np.mean(sel_scores[:, 1]), # dev score used for save checkpoint
    }
    return metrics


