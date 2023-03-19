from numpy import product
import torch
import os
import os
import numpy as np
import itertools
import regex as re

from pathlib import Path
from common.dataset import CustomDataset



def load_raw_dataset(dataset_name, set_name, partition=None, return_offsets=False, load_shuffle=False):
    """
        Load from the specified dataset. Note that the data path is hard-coded here!
    Args:
        dataset_name: the name of the dataset
        set_name: the set to load (train, val, test)
    Returns:
        ids: the ids of the examples
        sources: the list of sources
        targets: the list of targets
        offsets(Optional): the offsets of the cut points
    """
    assert partition is None or partition in ['1_half', '2_half', 'full']
    print(f"Loading {set_name} set from {dataset_name} dataset...")
    print(f"Using partition {partition}...")
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    if load_shuffle:
        dataset_folder = cur_folder.parent.parent / 'data' / 'raw_shuffled' / dataset_name
    else:
        dataset_folder = cur_folder.parent.parent / 'data' / 'raw_not_shuffled' / dataset_name
    file_path = dataset_folder / f'{set_name}.jsonl'
    ds = CustomDataset.from_jsonl(file_path)
    ids = [item['id'] for item in ds]
    sources = [item['source'] for item in ds]
    targets = [item['target'] for item in ds]
    offsets = (0, len(sources))
    if partition in ['1_half', '2_half']:
        if partition == '1_half':
            offsets = (0, len(sources)//2)
            ids = ids[:len(sources) // 2]
            sources = sources[:len(sources) // 2]
            targets = targets[:len(targets) // 2]
        else:
            offsets = (len(sources)//2, len(sources))
            ids = ids[len(sources) // 2:]
            sources = sources[len(sources) // 2:]
            targets = targets[len(targets) // 2:]
    if return_offsets:
        return ids, sources, targets, offsets
    else:
        return ids, sources, targets


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
    ids = [i for i in range(len(sources))]
    sources = [s.replace("\n", " ").replace("\t", " ").replace("\r", " ") for s in sources]
    targets = [t.replace("\n", " ").replace("\t", " ").replace("\r", " ") for t in targets]
    if shuffle:
        dataset_folder = cur_folder.parent.parent / 'data' / 'raw_shuffled' / dataset_name
        print("Shuffling the dataset...")
        indices = np.random.permutation(len(sources))
        print(indices[:10])
        sources = [sources[i] for i in indices]
        targets = [targets[i] for i in indices]
        ids = [ids[i] for i in indices]
    if isinstance(max_size, int) and max_size > 0:
        dataset_folder = cur_folder.parent.parent / 'data' / 'raw_not_shuffled' / dataset_name
        sources = sources[:max_size]
        targets = targets[:max_size]
        ids = ids[:max_size]
    file_path = dataset_folder / f'{set_name}.jsonl'
    ds = CustomDataset.from_raw(sources, targets, ids=ids)
    ds.to_jsonl(file_path)
    print(f"Saved rawdataset {dataset_name} to {file_path}.")

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
    print(f"Saved candidates to {pkl_path / f'candidates{postfix}.pkl{shard_postfix}'}")

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
    print(f"Saved the candidate scores to {pkl_path / f'cand_scores_{model_name}_{metric_name}.pkl'}")

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
    print(f"Saved the data to pkl files:", set_path / f"sources.pkl{shard_postfix}", set_path / f"targets.pkl{shard_postfix}")

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
        models.sort()
    if generation_methods == None:
        generation_methods = list(set([t[1] for t in cur_types]))
        generation_methods.sort()
    for model, generation_method in itertools.product(models, generation_methods):
        if (model, generation_method) not in cur_types:
            print(f"Warning: model: {model}, generation method: {generation_method} candidates not found")
            continue
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
    print(f"Saved the prepared dataset to {ds_path}")

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
            if re.match(r'candidates_[a-zA-Z0-9_]+.pkl.\d+_\d+', file.name):
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

def _get_all_parallel_idxs(file):
    """
        Get the start and end parallel indices
        i.e. find the files that apply "{file}.{start_idx}_{end_idx}"
    Args:
        file: the file name
    Returns:
        start_idxs: the start indices
        end_idxs: the end indices
        same length
    """
    if not isinstance(file, Path):
        file = Path(file)
    idxs = {}
    for f in file.parent.iterdir():
        if re.match(f'{file.name}.\d+_\d+', f.name):
            start_idx, end_idx = f.name.split('.')[-1].split('_')
            idxs[int(start_idx)] = int(end_idx)
    start_idxs = sorted(list(idxs.keys()))
    end_idxs = [idxs[start_idx] for start_idx in start_idxs]
    assert start_idxs[1:] == end_idxs[:-1], f"Start and end indices do not match: {start_idxs} {end_idxs}"
    return start_idxs, end_idxs

def _get_sub_parallel_idxs(start_idx, end_idx, start_idxs, end_idxs):
    """
        get the parallel indices that cover the range [start_idx, end_idx]
    Args:
        start_idx: the start index
        end_idx: the end index
        start_idxs: the list of start indices
        end_idxs: the list of end indices
    Returns:
        sub_start_idxs: the start indices that cover the range
        sub_end_idxs: the end indices that cover the range
    """
    idxs = dict(zip(start_idxs, end_idxs))
    start_idxs = sorted(list(idxs.keys()))
    end_idxs = [idxs[start_idx] for start_idx in start_idxs]
    left, right = None
    for i in range(len(start_idxs)-1, -1, -1):
        if start_idx >= start_idxs[i]:
            left = i
    for i in range(len(end_idxs)):
        if end_idx < end_idxs[i]:
            right = i
            break
    assert left is not None and right is not None, "The start and end indices lie outside the range of the list"
    sub_start_idxs = start_idxs[left:right+1]
    sub_end_idxs = end_idxs[left:right+1]
    # check the continuity
    assert sub_start_idxs[1:] == sub_end_idxs[:-1], f"Start and end indices are not continuous: {sub_start_idxs} {sub_end_idxs}"
    return sub_start_idxs, sub_end_idxs


def _save_pkl_parallel(file, obj, start_idx, end_idx):
    """
        Save the object to a pickle file
    Args:
        file: the file name
        start_idx: the start index
        end_idx: the end index
        obj: the object
    """
    if not isinstance(file, Path):
        file = Path(file)
    torch.save(obj, f'{file}.{start_idx}_{end_idx}')

def _load_pkl_parallel(file, start_idx, end_idx):
    """
        Load the parallel contents from the file
    Args:
        file: the file name
        start_idx: the start index
        end_idx: the end index

    """
    if not isinstance(file, Path):
        file = Path(file)
    start_idxs, end_idxs = _get_all_parallel_idxs(file)
    sub_start_idxs, sub_end_idxs = _get_sub_parallel_idxs(start_idx, end_idx, start_idxs, end_idxs)
    contents = []
    for start_idx, end_idx in zip(sub_start_idxs, sub_end_idxs):
        shard_postfix = f'.{start_idx}_{end_idx}'
        shard_file = file.parent / (file.name + shard_postfix)
        sub_contents = torch.load(shard_file)
        contents.extend(sub_contents)
    # clip the contents into the range
    contents = contents[start_idx-sub_start_idxs[0]:end_idx-sub_start_idxs[0]+1]

def save_pkl(file, obj, start_idx=None, end_idx=None):
    """
        Save the object to a pickle file
    Args:
        file: the file name
        obj: the object
    """
    if not isinstance(file, Path):
        file = Path(file)
    if start_idx is not None and end_idx is not None:
        _save_pkl_parallel(file, obj, start_idx, end_idx)
    else:
        torch.save(obj, file)

def load_pkl(file, start_idx=None, end_idx=None):
    """
        Load a pickle file
    Args:
        file: the file name
    Returns:
        the loaded object
    """
    if not isinstance(file, Path):
        file = Path(file)
    if start_idx is not None and end_idx is not None:
        return _load_pkl_parallel(file, start_idx, end_idx)
    else:
        return torch.load(file)
