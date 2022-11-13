
import psutil
import os
from evaluate import load
from sacrebleu import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu
from typing import List, Optional, Union, Dict, Tuple
from absl import logging
from torch import split
from tqdm import tqdm
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from tqdm.contrib.concurrent import process_map
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
logging.set_verbosity(logging.WARNING)


SUPPORTED_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu', 'bleurt', "cider", "spice"]
METRIC_WEIGHTS = {
    "rouge1": 1.0,
    "rouge2": 1.0,
    "rougeL": 1.0,
    "rougeLsum": 1.0,
    "bleu": 0.01,
    "bleurt": 1.0,
    "cider": 0.01,
    "spice": 0.01,
} # scale to 0-1

def pre_rouge_processing(summary):
    summary = summary.replace("<n>", " ")
    summary = "\n".join(sent_tokenize(summary))
    return summary

def eval_rouge(
    hypotheses: Union[List[List[str]], List[str]],
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
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]
    assert set(rouge_types) <= set(['rouge1', 'rouge2', 'rougeL', 'rougeLsum']), "Rouge types should be in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']"
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    rouge_scores = {rouge_type: [[] for _ in range(len(hypotheses))] for rouge_type in rouge_types}
    with tqdm(total=len(hypotheses), desc="Evaluating rouge") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            for hypo in hypo_group:
                scores = scorer.score(ref, pre_rouge_processing(hypo))
                for rouge_type in rouge_types:
                    rouge_scores[rouge_type][i].append(scores.get(rouge_type).fmeasure)
            pbar.update(1)
    # nested remove list with single element
    if all([all([len(score) == 1 for score in scores]) for scores in rouge_scores.values()]):
        rouge_scores = {rouge_type: [score[0] for score in scores] for rouge_type, scores in rouge_scores.items()}
    return rouge_scores

def eval_bleu(
    hypotheses: Union[List[List[str]], List[str]],
    references: List[str]
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    assert len(hypotheses) == len(references), f"Length of hypotheses {len(hypotheses)} and references {len(references)} should be the same."
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]
    bleu_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleu") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleu_scores.append([])
            for hypo in hypo_group:
                bleu_scores[i].append(sentence_bleu(hypo, [ref]).score)
            pbar.update(1)
    # nested remove list with single element
    if all([len(score) == 1 for score in bleu_scores]):
        bleu_scores = [score[0] for score in bleu_scores]
    return bleu_scores

def eval_bleurt(
    hypotheses: Union[List[List[str]], List[str]],
    references: List[str]
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    assert len(hypotheses) == len(references)
    bleurt_scorer = load('bleurt')
    bleurt_scores = []
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]
    with tqdm(total=len(hypotheses), desc="Evaluating bleurt") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleurt_scores.append([])
            for hypo in hypo_group:
                result = bleurt_scorer.compute(predictions=[hypo], references=[ref])
                bleurt_scores[i].append(result['scores'][0])
            pbar.update(1)
    # nested remove list with single element
    if all([len(score) == 1 for score in bleurt_scores]):
        bleurt_scores = [score[0] for score in bleurt_scores]
    return bleurt_scores

def eval_cider(
    hypotheses: Union[List[List[str]], List[str]],
    references: List[str]
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    print("Evaluating cider")
    assert len(hypotheses) == len(references)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]
    cider_scorer = Cider()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = [{"caption": ref}]
            res[id] = [{"caption": hypo}]
            hypo_ids_per_ref[i].append(id)
            id += 1
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    score, scores = cider_scorer.compute_score(gts, res)
    cider_scores = [[scores[hypo_id]*10 for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    # nested remove list with single element
    if all([len(score) == 1 for score in cider_scores]):
        cider_scores = [score[0] for score in cider_scores]
    return cider_scores

def eval_spice(
    hypotheses: Union[List[List[str]], List[str]],
    references: List[str]
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    print("Evaluating spice")
    assert len(hypotheses) == len(references)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]
    spice_scorer = Spice()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = [{"caption": ref}]
            res[id] = [{"caption": hypo}]
            hypo_ids_per_ref[i].append(id)
            id += 1
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    score, scores = spice_scorer.compute_score(gts, res)
    spice_scores = [[scores[hypo_id]['All']['f'] for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    # nested remove list with single element
    if all([len(score) == 1 for score in spice_scores]):
        spice_scores = [score[0] for score in spice_scores]
    return spice_scores


def _overall_eval_multi_process(data):
    candidates, targets, metrics = data
    s = psutil.Process(os.getpid())
    cpu_id = s.cpu_num()
    print("Worker {} is evaluating".format(cpu_id))
    return overall_eval(candidates, targets, metrics)

def _overall_eval(candidates, targets, metrics:List[str]):
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
    if 'cider' in metrics:
        cider_scores = eval_cider(candidates, targets)
        scores.update({'cider': cider_scores})
    if 'spice' in metrics:
        spice_scores = eval_spice(candidates, targets)
        scores.update({'spice': spice_scores})
    return scores

def overall_eval(
    candidates:Union[List[List[str]], List[str]],
    targets: List[str],
    metrics:List[str],
    num_workers:int=1
    ) -> Dict[str, List[float]]:
    """
    Args:
        candidates: the candidates
        targets: the targets
        metrics: the metrics to be evaluated
        num_workers: the number of workers to be used
    Return:
        A dict of scores, same shape with candidates for each metric
    """
    if num_workers > 1:
        cpu_num = psutil.cpu_count(logical=False)
        num_workers = min(num_workers, cpu_num)
        print("Using {} workers to evaluate".format(num_workers))
        chunk_size = len(candidates) // num_workers + 1
        candidates_chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        targets_chunks = [targets[i:i + chunk_size] for i in range(0, len(targets), chunk_size)]
        datas = [(candidates_chunks[i], targets_chunks[i], metrics) for i in range(len(candidates_chunks))]
        scores_chunks = process_map(_overall_eval_multi_process, datas, chunksize=1, max_workers=num_workers)
        scores = {}
        for chunk in scores_chunks:
            for k, v in chunk.items():
                scores[k] = scores.get(k, []) + v
    else:
        scores = _overall_eval(candidates, targets, metrics)
    return scores


