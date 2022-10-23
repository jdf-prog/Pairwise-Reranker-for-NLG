import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import EvalPrediction
from typing import Dict, List, Optional, Tuple, Union, Any
from common.evaluation import (
    eval_rouge,
    eval_bleu,
)
class RerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return super().compute_loss(model, inputs, return_outputs)

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        return super().compute_metrics(eval_pred)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    predictions, labels = eval_pred
    metrics = {}
    # evaluate rouge
    rouge_scores = eval_rouge(predictions, labels)
    for rouge_type, scores in rouge_scores.items():
        rouge_scores[rouge_type] = np.mean(scores)
    metrics.update(rouge_scores)
    return metrics
