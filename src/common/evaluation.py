
import psutil
import os
import numpy as np
import spacy
from evaluate import load
from sacrebleu import sentence_bleu, corpus_bleu
from nltk import word_tokenize
from typing import List, Optional, Union, Dict, Tuple
from absl import logging
from torch import split
from tqdm import tqdm
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from tqdm.contrib.concurrent import process_map
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
logging.set_verbosity(logging.WARNING)


SUPPORTED_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu', 'bleurt', "cider", "spice", "bleu4"]
METRIC_WEIGHTS = {
    "rouge1": 1.0,
    "rouge2": 1.0,
    "rougeL": 1.0,
    "rougeLsum": 1.0,
    "bleu": 0.01,
    "bleu4": 0.01,
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
    do_flatten = False
    assert len(hypotheses) == len(references)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            do_flatten = True
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
    if do_flatten:
        assert all([all([len(score) == 1 for score in scores]) for scores in rouge_scores.values()])
        rouge_scores = {rouge_type: [score[0] for score in scores] for rouge_type, scores in rouge_scores.items()}
    return rouge_scores

def eval_bleu(
    hypotheses: Union[List[List[str]], List[str]],
    references: List[str],
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    do_flatten = False
    assert len(hypotheses) == len(references), f"Length of hypotheses {len(hypotheses)} and references {len(references)} should be the same."
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            do_flatten = True
            hypotheses[i] = [hypotheses[i]]
    bleu_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleu") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleu_scores.append([])
            for hypo in hypo_group:
                bleu_scores[i].append(sentence_bleu(hypo, [ref]).score)
            pbar.update(1)
    # nested remove list with single element
    if do_flatten:
        assert all([len(score) == 1 for score in bleu_scores])
        bleu_scores = [score[0] for score in bleu_scores]
    return bleu_scores

def eval_bleu4(
    hypotheses: Union[List[List[str]], List[str]],
    references: List[str],
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    print("Evaluating bleu4")
    do_flatten = False
    assert len(hypotheses) == len(references)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            do_flatten = True
            hypotheses[i] = [hypotheses[i]]

    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join([token.text for token in nlp(hypotheses[i][j])])
        references[i] = " ".join([token.text for token in nlp(references[i])])

    bleu4_scorer = Bleu(4)
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = [ref]
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = bleu4_scorer.compute_score(gts, res)
    for method in zip(("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"), score):
        print("%s: %0.3f" % method)
    bleu4_scores = scores[3]
    bleu4_scores = [[bleu4_scores[hypo_id]*100 for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    # nested remove list with single element
    if do_flatten:
        assert all([len(score) == 1 for score in bleu4_scores])
        bleu4_scores = [score[0] for score in bleu4_scores]
    return bleu4_scores

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
    do_flatten = False
    assert len(hypotheses) == len(references)
    bleurt_scorer = load('bleurt')
    bleurt_scores = []
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            do_flatten = True
            hypotheses[i] = [hypotheses[i]]
    with tqdm(total=len(hypotheses), desc="Evaluating bleurt") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleurt_scores.append([])
            for hypo in hypo_group:
                result = bleurt_scorer.compute(predictions=[hypo], references=[ref])
                bleurt_scores[i].append(result['scores'][0])
            pbar.update(1)
    # nested remove list with single element
    if do_flatten:
        assert all([len(score) == 1 for score in bleurt_scores])
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
    do_flatten = False
    assert len(hypotheses) == len(references)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            do_flatten = True
            hypotheses[i] = [hypotheses[i]]

    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join([token.text for token in nlp(hypotheses[i][j])])
        references[i] = " ".join([token.text for token in nlp(references[i])])

    cider_scorer = Cider()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = [ref]
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = cider_scorer.compute_score(gts, res)
    cider_scores = [[scores[hypo_id]*10 for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    # nested remove list with single element
    if do_flatten:
        assert all([len(score) == 1 for score in cider_scores])
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
    do_flatten = False
    assert len(hypotheses) == len(references)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            do_flatten = True
            hypotheses[i] = [hypotheses[i]]
    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join([token.text for token in nlp(hypotheses[i][j])])
        references[i] = " ".join([token.text for token in nlp(references[i])])

    spice_scorer = Spice()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0
    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = [ref]
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = spice_scorer.compute_score(gts, res)
    spice_scores = [[scores[hypo_id]['All']['f']*100.0 for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    # nested remove list with single element
    if do_flatten:
        assertall([len(score) == 1 for score in spice_scores])
        spice_scores = [score[0] for score in spice_scores]
    return spice_scores



def compute_new_n_gram(source:str, candidate:str):
    """
        computer the new n-grams in the candidate compared to source text
    """
    # text
    text = source.lower()
    text_words = word_tokenize(text)
    text_bigrams = [[text_words[j], text_words[j + 1]] for j in range(len(text_words) - 1)]
    text_trigrams = [[text_words[j], text_words[j + 1], text_words[j + 2]] for j in range(len(text_words) - 2)]
    text_quadrigrams = [[text_words[j], text_words[j + 1], text_words[j + 2], text_words[j + 3]] for j in range(len(text_words) - 3)]

    # candidate
    candidate = candidate.lower().replace("<n>", " ")
    candidate_words = word_tokenize(candidate)

    unigrams, bigrams, trigrams, quadrigrams = 0, 0, 0, 0
    for j in range(len(candidate_words)):
        if not(candidate_words[j] in text_words):
            unigrams += 1
        if j < len(candidate_words) - 1:
            bigram = [candidate_words[j], candidate_words[j + 1]]
            if not(bigram in text_bigrams):
                bigrams += 1
        if j < len(candidate_words) - 2:
            trigram = [candidate_words[j], candidate_words[j + 1], candidate_words[j + 2]]
            if not(trigram in text_trigrams):
                trigrams += 1
        if j < len(candidate_words) - 3:
            quadrigram = [candidate_words[j], candidate_words[j + 1], candidate_words[j + 2], candidate_words[j + 3]]
            if not(quadrigram in text_quadrigrams):
                quadrigrams += 1
    new_unigram, new_bigram, new_trigram, new_quadrigram = 0, 0, 0, 0
    if len(candidate_words) > 0:
        new_unigram = unigrams / (len(candidate_words) - 0)
    if len(candidate_words) > 1:
        new_bigram = bigrams / (len(candidate_words) - 1)
    if len(candidate_words) > 2:
        new_trigram = trigrams / (len(candidate_words) - 2)
    if len(candidate_words) > 3:
        new_quadrigram = quadrigrams / (len(candidate_words) - 3)
    return new_unigram, new_bigram, new_trigram, new_quadrigram


def eval_novel_n_gram(
    sources: List[str],
    hypotheses: Union[List[List[str]], List[str]],
    ) -> List[float]:
    """
        evaluate the novel n-gram in the hypotheses compared to the origianl soiurce
    """
    print("Evaluating novel n-gram")
    assert len(hypotheses) == len(sources)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]

    new_unigrams, new_bigrams, new_trigrams, new_quadrigrams = [], [], [], []
    for i, (source, hypo_group) in tqdm(enumerate(zip(sources, hypotheses)), desc="evaluate novel n-grams"):
        new_unigrams.append([])
        new_bigrams.append([])
        new_trigrams.append([])
        new_quadrigrams.append([])
        for hypo in hypo_group:
            new_unigram, new_bigram, new_trigram, new_quadrigram = \
                compute_new_n_gram(source, hypo)
            new_unigrams[i].append(new_unigram)
            new_bigrams[i].append(new_bigram)
            new_trigrams[i].append(new_trigram)
            new_quadrigrams[i].append(new_quadrigram)

    new_unigrams = np.array(new_unigrams)
    m_uni = 100 * np.mean(new_unigrams)
    new_bigrams = np.array(new_bigrams)
    m_bi = 100 * np.mean(new_bigrams)
    new_trigrams = np.array(new_trigrams)
    m_tri = 100 * np.mean(new_trigrams)
    new_quadrigrams = np.array(new_quadrigrams)
    m_quadri = 100 * np.mean(new_quadrigrams)
    print("New unigrams: {:.2f}, bigrams: {:.2f}, trigrams: {:.2f}, quadrigrams: {:.2f}".format(m_uni, m_bi, m_tri, m_quadri))
    # nested remove list with single element
    if all([len(score) == 1 for score in new_unigrams]):
        new_unigrams = [score[0] for score in new_unigrams]
    if all([len(score) == 1 for score in new_bigrams]):
        new_bigrams = [score[0] for score in new_bigrams]
    if all([len(score) == 1 for score in new_trigrams]):
        new_trigrams = [score[0] for score in new_trigrams]
    if all([len(score) == 1 for score in new_quadrigram]):
        new_quadrigram = [score[0] for score in new_quadrigram]
    return new_unigrams, new_bigrams, new_trigrams, new_quadrigrams

def eval_distinct_n_grams(texts:Union[List[List[str]], List[str]]):
    print("evaluating distinct n-grams")
    for i in range(len(texts)):
        if isinstance(texts[i], str):
            texts[i] = [texts[i]]

    uni_unigrams, uni_bigrams, uni_trigrams, uni_quadrigrams = [], [], [], []
    for i, text_group in tqdm(enumerate(texts), desc='evaluting distinct n-grams'):
        unigrams = []
        bigrams = []
        trigrams = []
        quadrigrams = []
        for text in text_group:
            text = text.lower()
            text_words = word_tokenize(text)
            text_bigrams = [(text_words[j], text_words[j + 1]) for j in range(len(text_words) - 1)]
            text_trigrams = [(text_words[j], text_words[j + 1], text_words[j + 2]) for j in range(len(text_words) - 2)]
            text_quadrigrams = [(text_words[j], text_words[j + 1], text_words[j + 2], text_words[j + 3]) for j in range(len(text_words) - 3)]
            unigrams.extend(text_words)
            bigrams.extend(text_bigrams)
            trigrams.extend(text_trigrams)
            quadrigrams.extend(text_quadrigrams)
        unigrams = set(unigrams)
        bigrams = set(unigrams)
        trigrams = set(trigrams)
        quadrigrams = set(quadrigrams)
        uni_unigrams.append(len(unigrams))
        uni_bigrams.append(len(bigrams))
        uni_trigrams.append(len(trigrams))
        uni_quadrigrams.append(len(quadrigrams))
    print(f"Mean unique 1-grams: {np.mean(uni_unigrams)}")
    print(f"Mean unique 2-grams: {np.mean(uni_bigrams)}")
    print(f"Mean unique 3-grams: {np.mean(uni_trigrams)}")
    print(f"Mean unique 4-grams: {np.mean(uni_quadrigrams)}")
    return uni_unigrams, uni_bigrams, uni_trigrams, uni_quadrigrams

def eval_self_bleu(texts:List[List[str]]):
    print("evaluating self bleu")
    for i in range(len(texts)):
        assert isinstance(texts[i], list)

    self_bleus = []
    for i, text_group in tqdm(enumerate(texts), desc='evaluting distinct n-grams'):
        group_self_bleus = []
        for j in range(len(text_group)):
            hypo = text_group[j]
            refs = text_group[:j] + text_group[j+1:]
            group_self_bleus.append(sentence_bleu(hypothesis=hypo, references=refs).score)
        self_bleus.append(np.mean(group_self_bleus))
    print(f"self BLEUs mean: {np.mean(self_bleus)}")
    return self_bleus

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
    if 'bleu4' in metrics:
        bleu4_scores = eval_bleu4(candidates, targets)
        scores.update({'bleu4': bleu4_scores})
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


