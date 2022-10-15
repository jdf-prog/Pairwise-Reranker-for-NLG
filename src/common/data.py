import torch
import os
import os
import torch

from pathlib import Path
from common.dataset import CustomDataset



def load_raw_dataset(dataset_name, set_name):
    """
        Load from the specified dataset. Note that the data path is hard-coded here!
    Args:
        dataset_name: the name of the dataset
        set_name: the set to load (train, val, test)
    Returns:
        sources: the list of sources
        targets: the list of targets
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    dataset_folder = cur_folder.parent.parent / 'data' / 'raw' / dataset_name
    file_path = dataset_folder / f'{set_name}.jsonl'
    ds = CustomDataset.from_jsonl(file_path)
    sources = [item['source'] for item in ds]
    targets = [item['target'] for item in ds]
    return sources, targets


def save_raw_dataset(dataset_name, set_name, sources, targets):
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
    ds = CustomDataset.from_raw(sources, targets)
    ds.to_jsonl(file_path)

def save_pkl_data(dataset_name, set_name, generation_method, model_name, sources, candidates, targets):
    """
        Save the data to pkl files.
        Note that the data path is hard-coded here!
    Args:
        data_path: the path to save the data
        sources: the list of original sentences, [s1, s2, ...]
        candidates: the list of candidates, [[c1_1, c1_2, c1_3], [c2_1, c2_2, c2_3], ...]
        targets: the list of targets, [t1, t2, ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    num_candidates = len(candidates[0])

    print("Saving the data to pkl files...")
    pkl_path.mkdir(parents=True, exist_ok=True)
    postfix = f"_{model_name}"
    torch.save(sources, pkl_path / f"sources.pkl")
    torch.save(candidates, pkl_path / f"candidates{postfix}.pkl")
    torch.save(targets, pkl_path / f"targets.pkl")
    assert len(sources) == len(candidates) == len(targets)
    print("Saved the data to pkl files!")

def load_pkl_data(dataset_name, set_name, generation_method, model_name):
    """
        Load the data from pkl files.
        Note that the data path is hard-coded here!
    Args:
        data_path: the path to load the data
    Returns:
        sources: the list of original sentences, [s1, s2, ...]
        candidates: the list of candidates, [[c1_1, c1_2, c1_3], [c2_1, c2_2, c2_3], ...]
        targets: the list of targets, [t1, t2, ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    postfix = f"_{model_name}"
    sources = torch.load(pkl_path / f"sources.pkl")
    candidates = torch.load(pkl_path / f"candidates{postfix}.pkl")
    targets = torch.load(pkl_path / f"targets.pkl")
    assert len(sources) == len(candidates) == len(targets)
    return sources, candidates, targets

def load_pkl_candidates(dataset_name, set_name, generation_method, model_name):
    """
        Load the candidates from pkl files.
        Note that the data path is hard-coded here!
    Args:
        data_path: the path to load the data
    Returns:
        candidates: the list of candidates, [[c1_1, c1_2, c1_3], [c2_1, c2_2, c2_3], ...]
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    pkl_path = cur_folder.parent.parent / 'data' / dataset_name / set_name / generation_method
    postfix = f"_{model_name}"
    candidates = torch.load(pkl_path / f"candidates{postfix}.pkl")
    return candidates

def load_prepared_dataset(dataset_name, set_name) -> CustomDataset:
    """
        Load the computed and prepared dataset
        Note that the data path is hard-coded here!
    """
    cur_folder = Path(os.path.realpath(os.path.dirname(__file__)))
    ds_path = cur_folder.parent.parent / 'data' / 'prepared' / dataset_name / set_name / 'dataset.jsonl'
    if not ds_path.exists():
        sources, targets = load_raw_dataset(dataset_name, set_name)
        ds = CustomDataset.from_raw(sources, targets)
    else:
        ds = CustomDataset.from_jsonl(ds_path)
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
        for model_name in generation_method.iterdir():
            if (
                not model_name.is_file() or
                not model_name.name.endswith('.pkl') or
                not model_name.name.startswith('candidates')
            ):
                continue
            model = model_name.stem[len('candidates_'):]
            candidate_types.append((model, generation_method.name))
    return candidate_types
