import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Sampler, SubsetRandomSampler
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (
    PredictionOutput,
    IntervalStrategy,
)
from tqdm import tqdm
from transformers.trainer import TrainingArguments, TrainerState, TrainerControl
from typing import Optional, Sequence, Iterator, Sized


class CurriculumDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_candidates=None):
        self.data = data
        self.n_candidates = n_candidates if n_candidates is not None and n_candidates > 0 else None
        self.n_tasks = len(self.data[0]['candidates'][0]['scores']) if 'candidates' in self.data[0] else -1
        self.index_to_data_index = []
        for i in range(len(data)):
            for j in range(len(data[i]['candidates'])):
                for k in range(j + 1, len(data[i]['candidates'])):
                    self.index_to_data_index.append((i, j, k))
        # subsampling, reduce the amount from O(n^2) to O(n)
        subsampling_num = len(self.data) * n_candidates
        self.index_to_data_index = np.array(self.index_to_data_index)
        np.random.shuffle(self.index_to_data_index)
        self.index_to_data_index = self.index_to_data_index[:int(subsampling_num)]


    def __len__(self):
        return len(self.index_to_data_index)

    def __getitem__(self, index):
        i,j,k = self.index_to_data_index[index]
        data = {
            'index' : index,
            'source' : self.data[i]['source'],
            'target' : self.data[i]["target"],
            'candidate1' : self.data[i]['candidates'][j]['text'],
            'candidate2' : self.data[i]['candidates'][k]['text'],
            'score1' : torch.tensor([float(score) for score in self.data[i]['candidates'][j]['scores'].values()]),
            'score2' : torch.tensor([float(score) for score in self.data[i]['candidates'][k]['scores'].values()]),
        }
        return data


class CurriculumSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]
    def __init__(
        self,
        data_source: Sized,
        num_curriculum: int,
        curriculum_size: int=None,
        generator=None,
        ) -> None:
        self.data_source = data_source
        self.generator = generator
        self.num_curriculum = num_curriculum
        self.curriculum_size = len(self.data_source) // self.num_curriculum if curriculum_size is None else curriculum_size
        self.num_curriculum_learned = 0
        self.shuffled_data_idx = np.arange(0, len(self.data_source))
        np.random.shuffle(self.shuffled_data_idx)
        self.indices = np.arange(0, self.curriculum_size)
        self.indices = self.shuffled_data_idx[self.indices]

        # 10 curriculum, 10000, 1000,


    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

    def update_indices(self, indices):
        indices = np.array(indices)
        self.indices = indices

    def get_next_curriculum_indices(self):
        if self.num_curriculum_learned * self.curriculum_size < len(self.data_source):
            indices = np.arange(
                self.curriculum_size * self.num_curriculum_learned,
                min(self.curriculum_size * (self.num_curriculum_learned + 1), len(self.data_source)))
            indices = self.shuffled_data_idx[indices]
            self.num_curriculum_learned += 1
            return indices
        else:
            self.num_curriculum_learned = 0
            indices = np.arange(0, self.curriculum_size)
            indices = self.shuffled_data_idx[indices]
            return indices



class CurriculumCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    This callback is modified to support curriculum learning.
    """

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        **kwargs):
        # Adjust the next epoch's data according to the model performance
        if args.curriculum_learning:
            # print("Adjusting Curriculum")
            # print("Curriculum indices: ", train_dataloader.sampler.indices[:100])
            next_curriculum_indices = train_dataloader.sampler.get_next_curriculum_indices()
            # print("Next Curriculum indices: ", next_curriculum_indices[:100])
            next_curriculum_data = []
            for i in next_curriculum_indices:
                next_curriculum_data.append(train_dataloader.dataset[i])
            dataloader = DataLoader(
                next_curriculum_data,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=train_dataloader.collate_fn,
                shuffle=False
            )
            model.eval()
            preds = []
            dif_scores = []
            device = next(model.parameters()).device

            with tqdm(total=len(dataloader), desc="Selecting next curriculum") as progress_bar:
                for inputs in dataloader:
                    with torch.no_grad():
                        assert inputs['scores'].shape[1] == 2
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs)
                        pred = outputs['preds'].detach().cpu().numpy()
                        dif_score = inputs["scores"][:, 0] - inputs["scores"][:, 1]
                        dif_score = dif_score.detach().cpu().numpy()
                        preds.append(pred)
                        dif_scores.append(dif_score.sum(-1))
                        progress_bar.update(1)
            preds = np.concatenate(preds)
            dif_scores = np.concatenate(dif_scores)
            error_indices_flag = preds * dif_scores < 0
            # print("Error indices flag: ", error_indices_flag[:100])
            error_indices = np.where(error_indices_flag)[0]
            right_indices = np.where(~error_indices_flag)[0]
            # print("Error indices local: ", error_indices[:100])
            print("Select {}/{} indices for next curriculum".format(len(error_indices), len(next_curriculum_indices)))
            print("accuracy", np.mean(~error_indices_flag))
            error_indices = np.array(next_curriculum_indices)[error_indices]
            right_indices = np.array(next_curriculum_indices)[right_indices]
            selected_indices = np.concatenate([error_indices, right_indices[:len(right_indices) // 10]])
            # print("Error indices global: ", error_indices[:100])
            assert len(selected_indices.shape) == 1
            assert not set(selected_indices).intersection(set(train_dataloader.sampler.indices)), set(selected_indices).intersection(set(train_dataloader.sampler.indices))
            train_dataloader.sampler.update_indices(selected_indices)
        return control

def compute_metrics_for_curriculum(eval_pred):
    """
    Compute metrics for the model.
    Args:

    """
    preds, labels = eval_pred # scores [batch_size, n_candidates, n_tasks]
    pred_dif_scores = preds
    dif_scores = (labels[:, 0].sum(-1) - labels[:, 1].sum(-1))
    correct_flag = pred_dif_scores * dif_scores > 0
    acc = np.mean(correct_flag)

    metrics = {
        "acc": acc,
        "dev_score": acc,
    }

    return metrics

