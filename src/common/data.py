from numpy import product
import torch
import os
import os
import numpy as np
import itertools
import regex as re

from pathlib import Path
from common.dataset import CustomDataset



def load_raw_dataset(dataset_name, set_name, partition=None, return_offsets=False):
    """
        Load from the specified dataset. Note that the data path is hard-coded here!
    Args:
        dataset_name: the name of the dataset
        set_name: the set to load (train, val, test)
    Returns:
        sources: the list of sources
        targets: the list of targets
        offsets: the offsets of the cut points
    """
    assert set_name in ['train', 'val', 'test']
    assert partition is None or partition in ['1_half', '2_half', 'full']
    print(f"Loading {set_name} set from {dataset_name} dataset...")
    print(f"Using partition {partition}...")
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    dataset_folder = cur_folder.parent.parent / 'data' / 'raw' / dataset_name
    file_path = dataset_folder / f'{set_name}.jsonl'
    ds = CustomDataset.from_jsonl(file_path)
    sources = [item['source'] for item in ds]
    targets = [item['target'] for item in ds]
    offsets = (0, len(sources))
    if partition in ['1_half', '2_half']:
        if partition == '1_half':
            offsets = (0, len(sources)//2)
            sources = sources[:len(sources) // 2]
            targets = targets[:len(targets) // 2]
        else:
            offsets = (len(sources)//2, len(sources))
            sources = sources[len(sources) // 2:]
            targets = targets[len(targets) // 2:]
    if return_offsets:
        return sources, targets, offsets
    else:
        return sources, targets


def save_raw_dataset(dataset_name, set_name, sources, targets, shuffle=False, max_size=None):
    """
        Save the raw dataset to pkl files.
        Note that the data path is hard-coded here!
    Args:
        dataset_name: the name of the dataset
        set_name: the set to load (train, val, test)
        sources: the list of sources
        targets: the list of targets
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    dataset_folder = cur_folder.parent.parent / 'data' / 'raw' / dataset_name
    file_path = dataset_folder / f'{set_name}.jsonl'
    if shuffle:
        print("Shuffling the dataset...")
        indices = np.arange(len(sources))
        np.random.shuffle(indices)
        sources = [sources[i] for i in indices]
        targets = [targets[i] for i in indices]
    if isinstance(max_size, int) and max_size > 0:
        sources = sources[:max_size]
        targets = targets[:max_size]
    ds = CustomDataset.from_raw(sources, targets)
    ds.to_jsonl(file_path)

def exist_pkl_candidates(dataset_name, set_name, generation_method, model_name, start_idx=None, end_idx=None):
    """
        Check if the candidates pkl file exists.
        Note that the data path is hard-coded here!
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    postfix = f"_{model_name}"
    shard_postfix = f".{start_idx}_{end_idx}" if start_idx is not None and end_idx is not None else ""
    return (pkl_path / f"candidates{postfix}.pkl{shard_postfix}").exists()

def load_pkl_candidates(dataset_name, set_name, generation_method, model_name, start_idx=None, end_idx=None):
    """
        Load the candidates from pkl files.
        Note that the data path is hard-coded here!
    Returns:
        candidates: the list of candidates, [[c1_1, c1_2, c1_3], [c2_1, c2_2, c2_3], ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    postfix = f"_{model_name}"
    shard_postfix = f".{start_idx}_{end_idx}" if start_idx is not None and end_idx is not None else ""
    candidates = torch.load(pkl_path / f"candidates{postfix}.pkl{shard_postfix}")
    return candidates

def save_pkl_candidates(dataset_name, set_name, generation_method, model_name, candidates, start_idx=None, end_idx=None):
    """
        Save the candidates to pkl files.
        Note that the data path is hard-coded here!
    Args:
        candidates: the list of candidates, [[c1_1, c1_2, c1_3], [c2_1, c2_2, c2_3], ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    postfix = f"_{model_name}"
    pkl_path.mkdir(parents=True, exist_ok=True)
    shard_postfix = f".{start_idx}_{end_idx}" if start_idx is not None and end_idx is not None else ""
    torch.save(candidates, pkl_path / f"candidates{postfix}.pkl{shard_postfix}")

def load_pkl_cand_scores(dataset_name, set_name, generation_method, model_name, metric_name):
    """
        Load the candidates from pkl files.
        Note that the data path is hard-coded here!
    Returns:
        candidates: the list of candidates, [[c1_1, c1_2, c1_3], [c2_1, c2_2, c2_3], ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    scores = torch.load(pkl_path / f"cand_scores_{model_name}_{metric_name}.pkl")
    return scores

def save_pkl_cand_scores(dataset_name, set_name, generation_method, model_name, metric_name, scores):
    """
        Save the candidates to pkl files.
        Note that the data path is hard-coded here!
    Args:
        scores: the list of candidates, [[s1_1, s1_2, s1_3], [s2_1, s2_2, s2_3], ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    pkl_path.mkdir(parents=True, exist_ok=True)
    torch.save(scores, pkl_path / f"cand_scores_{model_name}_{metric_name}.pkl")

def load_pkl_sources_and_targets(dataset_name, set_name, start_idx=None, end_idx=None):
    """
        Load the data from pkl files.
        Note that the data path is hard-coded here!
    Returns:
        sources: the list of original sentences, [s1, s2, ...]
        targets: the list of targets, [t1, t2, ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    set_path = cur_folder.parent.parent / 'data' / dataset_name / set_name
    shard_postfix = f".{start_idx}_{end_idx}" if start_idx is not None and end_idx is not None else ""
    sources = torch.load(set_path / f"sources.pkl{shard_postfix}")
    targets = torch.load(set_path / f"targets.pkl{shard_postfix}")
    assert len(sources) == len(targets)
    return sources, targets

def save_pkl_sources_and_targets(dataset_name, set_name, sources, targets, start_idx=None, end_idx=None):
    """
        Save the sources and targets to pkl files.
        Note that the data path is hard-coded here!
    Args:
        sources: the list of original sentences, [s1, s2, ...]
        targets: the list of targets, [t1, t2, ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    set_path = cur_folder.parent.parent / 'data' / dataset_name / set_name
    set_path.mkdir(parents=True, exist_ok=True)
    shard_postfix = f".{start_idx}_{end_idx}" if start_idx is not None and end_idx is not None else ""
    torch.save(sources, set_path / f"sources.pkl{shard_postfix}")
    torch.save(targets, set_path / f"targets.pkl{shard_postfix}")
    assert len(sources) == len(targets)
    print("Saved the data to pkl files!")

def load_prepared_dataset(dataset_name, set_name, models:list=None, generation_methods:list=None, metrics:list=None) -> CustomDataset:
    """
        Load the computed and prepared dataset
        Note that the data path is hard-coded here!
    """
    cur_types = get_candidate_types(dataset_name, set_name)
    sources, targets = load_pkl_sources_and_targets(dataset_name, set_name)
    ds = CustomDataset.from_raw(sources, targets)

    if models == None:
        models = list(set([t[0] for t in cur_types]))
    if generation_methods == None:
        generation_methods = list(set([t[1] for t in cur_types]))
    for model, generation_method in itertools.product(models, generation_methods):
        assert (model, generation_method) in cur_types, f"{model} {generation_method} not in {cur_types}"
        cur_metrics = get_candidate_metrics(dataset_name, set_name, model, generation_method)
        candidates = load_pkl_candidates(dataset_name, set_name, generation_method, model)
        scores = {}
        if metrics == None:
            to_load_metrics = cur_metrics
        else:
            to_load_metrics = metrics
        for metric in to_load_metrics:
            assert metric in cur_metrics, f"{metric} not in {cur_metrics}"
            cand_scores = load_pkl_cand_scores(dataset_name, set_name, generation_method, model, metric)
            scores[metric] = cand_scores
        ds.add_candidates(model, generation_method, candidates, scores)
    ds.self_check()
    return ds

def save_prepared_dataset(dataset_name, set_name, dataset: CustomDataset):
    """
        Save the computed and prepared dataset
        Note that the data path is hard-coded here!
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    ds_path = cur_folder.parent.parent / 'data' / 'prepared' / dataset_name / set_name / 'dataset.jsonl'
    dataset.to_jsonl(ds_path)

def get_candidate_types(dataset_name, set_name):
    """
        Get the tuples of (model, generation_method)
        for each generation and return
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name
    candidate_types = []
    for generation_method in pkl_path.iterdir():
        if not generation_method.is_dir():
            continue
        for file in generation_method.iterdir():
            if (
                not file.is_file() or
                not file.name.endswith('.pkl') or
                not file.name.startswith('candidates')
            ):
                continue
            model = file.stem[len('candidates_'):]
            candidate_types.append((model, generation_method.name))
    return candidate_types

def get_parallel_candidate_types(dataset_name, set_name):
    """
        Get the tuples of (model, generation_method)
        for each generation and return
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name
    candidate_types = []
    for generation_method in pkl_path.iterdir():
        if not generation_method.is_dir():
            continue
        for file in generation_method.iterdir():
            if re.match(r'candidates_[a-zA-Z_]+.pkl.\d+_\d+', file.name):
                print(f"Found parallel candidates: {file.name}")
                model = file.name.split('.')[0][len('candidates_'):]
                start_idx, end_idx = file.name.split('.')[-1].split('_')
                candidate_types.append((model, generation_method.name, int(start_idx), int(end_idx)))
    return candidate_types

def get_candidate_metrics(dataset_name, set_name, model_name, generation_method):
    """
        Get the tuples of (metric_name, metric_value)
        for each metric and return
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    candidate_metrics = []
    for file in pkl_path.iterdir():
        if (
            not file.is_file() or
            not file.name.endswith('.pkl') or
            not file.name.startswith(f'cand_scores_{model_name}_')
        ):
            continue
        metric = file.stem[len(f'cand_scores_{model_name}_'):]
        candidate_metrics.append(metric)
    return candidate_metrics
