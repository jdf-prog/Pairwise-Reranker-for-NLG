
import psutil
import os
from evaluate import load
from sacrebleu import sentence_bleu
from typing import List, Optional, Union, Dict, Tuple
from absl import logging
from torch import split
from tqdm import tqdm
from rouge_score import rouge_scorer
logging.set_verbosity(logging.WARNING)


SUPPORTED_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu', 'bleurt']

def eval_rouge(
    hypotheses: List[List[str]],
    references: List[str],
    rouge_types: List[str]=['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    ) -> Dict[str, float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
        rouge_types: the rouge types to be used.

    Returns:
        A dict of rouge scores.
        key is the rouge type, value is the rouge score, in same shape with hypotheses.
    """
    assert len(hypotheses) == len(references)
    assert set(rouge_types) <= set(['rouge1', 'rouge2', 'rougeL', 'rougeLsum']), "Rouge types should be in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']"
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True, split_summaries=True)
    rouge_scores = {rouge_type: [[] for _ in range(len(hypotheses))] for rouge_type in rouge_types}
    with tqdm(total=len(hypotheses), desc="Evaluating rouge") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            for hypo in hypo_group:
                scores = scorer.score(ref, hypo)
                for rouge_type in rouge_types:
                    rouge_scores[rouge_type][i].append(scores.get(rouge_type).fmeasure)
            pbar.update(1)
    return rouge_scores

def eval_bleu(hypotheses: List[List[str]], references: List[str]) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    assert len(hypotheses) == len(references)
    bleu_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleu") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleu_scores.append([])
            for hypo in hypo_group:
                bleu_scores[i].append(sentence_bleu(hypo, [ref]).score)
            pbar.update(1)
    return bleu_scores

def eval_bleurt(hypotheses: List[List[str]], references: List[str]) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    assert len(hypotheses) == len(references)
    bleurt_scorer = load('bleurt')
    bleurt_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleurt") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleurt_scores.append([])
            for hypo in hypo_group:
                result = bleurt_scorer.compute(predictions=[hypo], references=[ref])
                bleurt_scores[i].append(result['scores'][0])
            pbar.update(1)
    return bleurt_scores

def overall_eval_multi_process(data):
    candidates, targets, metrics = data
    s = psutil.Process(os.getpid())
    cpu_id = s.cpu_num()
    print("Worker {} is evaluating".format(cpu_id))
    return overall_eval(candidates, targets, metrics)

def overall_eval(candidates, targets, metrics:List[str]):
    scores = {}
    rouge_tyeps = [metric for metric in metrics if metric.startswith('rouge')]
    if rouge_tyeps:
        rouge_scores = eval_rouge(candidates, targets, rouge_types=rouge_tyeps)
        scores.update(rouge_scores)
    if 'bleu' in metrics:
        bleu_scores = eval_bleu(candidates, targets)
        scores.update({'bleu': bleu_scores})
    if 'bleurt' in metrics:
        bleurt_scores = eval_bleurt(candidates, targets)
        scores.update({'bleurt': bleurt_scores})
    return scores
