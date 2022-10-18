
from evaluate import load
from sacrebleu import sentence_bleu
from typing import List, Optional, Union, Dict, Tuple
from absl import logging
from tqdm import tqdm
logging.set_verbosity(logging.WARNING)
class Support(object):

    METRICS = ['bleu','bleurt', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    evaluators = {}

    @staticmethod
    def check_metric_support(metric):
        if metric not in Support.METRICS:
            raise ValueError("Metric {} is not supported".format(metric))

    @staticmethod
    def get_supported_metrics():
        return Support.METRICS

    @staticmethod
    def add_self_metric(name, metric):
        """
        You can add a slf design metric to the reranker.
        The function should be of form 'metric(hypotheses, references) -> List[float]'
        """
        if name in Support.METRICS:
            raise ValueError("Metric {} is already supported".format(name))
        Support.evaluators[name] = metric

    @staticmethod
    def evaluate(hypothesis:str, reference:str, metric='bleu') -> Tuple[float]:
        """
        Evaluate the hypothesis and reference using the metric.

        Args:
            hypothesis: the hypothesis
            reference: the reference
            metric: the metric to be used. Support hugging face evaluate metric modules.
        """
        if metric == 'bleu':
            return sentence_bleu(hypothesis, [reference]).score
        elif metric == 'bleurt':
            if 'bleurt' not in Support.evaluators:
                Support.evaluators['bleurt'] = load('bleurt')
            result = Support.evaluators['bleurt'].compute(predictions=[hypothesis], references=[reference])
            return result['scores'][0]
        elif metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            if 'rouge' not in Support.evaluators:
                Support.evaluators['rouge'] = load('rouge')
            result = Support.evaluators['rouge'].compute(predictions=[hypothesis], references=[reference], rouge_types=[metric])
            return result[metric]
        else:
            if metric not in Support.evaluators:
                raise ValueError("Metric {} is not supported".format(metric))
            return Support.evaluators[metric]([hypothesis], [reference])[0]

