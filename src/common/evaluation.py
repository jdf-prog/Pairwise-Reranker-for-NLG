import os
import psutil
from typing import Tuple, Union, List, Dict
from evaluate import load
from sacrebleu import sentence_bleu
from absl import logging
from tqdm import tqdm
logging.set_verbosity(logging.WARNING)

class Support(object):
    """
        a class specify the supports of the baselines
    """
    METRICS = ['bleu','bleurt', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bert_score']
    def __init__(self):
        self.evaluators = {}

    @staticmethod
    def check_metric_support(metric: Union[list, str]) -> None:
        if isinstance(metric, str):
            metric = [metric]
        metrics_not_supported = [m for m in metric if m not in Support.METRICS]
        if len(metrics_not_supported) > 0:
            raise ValueError(f"Metrics {metrics_not_supported} not supported")

    def evaluate(self, hypothesis:str, reference:str, metric='bleu') -> Tuple[float]:
        """
        Evaluate the hypothesis and reference using the metric.

        Args:
            hypothesis: the hypothesis
            reference: the reference
            metric: the metric to be used. Support hugging face evaluate metric modules.
        """
        s = psutil.Process(os.getpid())
        current_cpu_number = s.cpu_num()
        print("current cpu number is {}".format(current_cpu_number))
        if metric == 'bleu':
            return sentence_bleu(hypothesis, [reference]).score
        elif metric == 'bleurt':
            if 'bleurt' not in self.evaluators:
                self.evaluators['bleurt'] = load('bleurt')
            result = self.evaluators['bleurt'].compute(predictions=[hypothesis], references=[reference])
            return result['scores'][0]
        elif metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            if 'rouge' not in self.evaluators:
                self.evaluators['rouge'] = load('rouge')
            result = self.evaluators['rouge'].compute(predictions=[hypothesis], references=[reference], rouge_types=[metric])
            return result[metric]
        elif metric == 'bert_score':
            if 'bert_score' not in self.evaluators:
                self.evaluators['bert_score'] = load('bertscore')
            result = self.evaluators['bert_score'].compute(predictions=[hypothesis], references=[reference], lang='en')
            return result['f1'][0]
        elif metric in self.METRICS:
            return self.evaluators[metric]([hypothesis], [reference])[0]
        else:
            if metric not in self.evaluators:
                raise ValueError("Metric {} is not supported".format(metric))
            return self.evaluators[metric]([hypothesis], [reference])[0]
