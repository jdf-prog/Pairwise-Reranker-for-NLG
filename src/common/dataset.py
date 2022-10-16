import os
import numpy as np
import json
import os
import json
import prettytable as pt
import logging as log
import numpy as np
import uuid
import multiprocess as mp
import psutil

from tqdm import tqdm
from typing import Counter, List, Dict, Tuple, Union, Optional, Set
from collections import Counter
from common.evaluation import (
    Support
)
class CustomDataset:

    def __init__(self, items:dict={}, file:str=None):
        """
        Initialize the dataset

        Args:
            items: the items to be used
        """
        self.file = file
        self.items = items
        # set logger
        self._set_logger()
        self.self_check()

    def _prepare_metrics(self, metrics:List[str]=None):
        """
        prepare the dataset for reranking.
        This function computes all the metrics value of each candidate for each source.

        Args:
            metrics: the metrics to be used.
        """
        evaluator = Support()
        for item in tqdm(self.items, desc="Preparing metrics"):
            for cand in item['candidates']:
                for metric in metrics:
                    # only compute the metric if it is not computed yet
                    if metric not in cand['scores']:
                        cand['scores'][metric] = evaluator.evaluate(hypothesis=cand['text'], reference=item['target'], metric=metric)

    def prepare_metrics(self, metrics:List[str]=None):
        """
        prepare the dataset for reranking.
        This function computes all the metrics value of each candidate for each source.

        Args:
            metrics: the metrics to be used.
            num_workers: the number of workers to be used for computing the metrics.
        """
        self.logger.info('Preparing metrics...')
        # if metrics is not specified, use 'bleu' by default
        if metrics is None:
            self.logger.info('No metrics specified. End preparing')

        # prepare the hugging face metric evaluator
        for metric in metrics:
            Support.check_metric_support(metric)

        prepared_metrics = self.prepared_metrics
        metrics_to_prepare = set(metrics) - set(prepared_metrics)


        self.logger.info("Metrics {} already prepared.".format(prepared_metrics))
        self.logger.info("Metrics {} will be prepared.".format(metrics_to_prepare))
        # evaluate and record the metric values
        self._prepare_metrics(metrics=metrics_to_prepare)

        self.logger.info('Finish preparing metrics {}.'.format(metrics_to_prepare))
        self.logger.info('The current available metrics are {}'.format(self.prepared_metrics))


    def add_candidates(self, model:str, generation_method: str, candidates:Union[List[str], List[List[str]]]):
        """
        Add candidates from the model to the data.

        Args:
            model: the name of the model
            generation_method: the name of the generation method
            candidates: the list of candidate texts
        """
        assert len(candidates) == len(self.items), "Number of candidates must be equal to number of items: {} != {}".format(len(candidates), len(self.items))
        for item, cands in zip(self.items, candidates):
            if isinstance(cands, str):
                cands = [cands]
            for cand in cands:
                item['candidates'].append({
                    "model": model,
                    "generation_method": generation_method,
                    "text": cand,
                    "scores": {}
                })

    @classmethod
    def from_structured(cls, items:List[Dict]):
        """
        Initialize the Formatter from a structured list of items.

        Args:
            items: the list of items
        """
        ds = CustomDataset(items)
        ds.logger.info(f"Dataset initialized from structured data")
        return ds

    @classmethod
    def from_raw(
        cls,
        sources:List[str],
        targets:List[str],
        ids:list=None,
        ):
        """
        Initialize the Formatter from raw data.

        Args:
            sources: the list of original sentences
            targets: the list of targets
            Ids: the list of Ids. optional.
        """
        assert len(sources) == len(targets), "Number of original sentences and reference sentences must be equal"
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(sources))]
        else:
            assert len(ids) == len(sources), "Number of Ids must be equal to number of original sentences"
        items = []
        for id, source, target in zip(ids, sources, targets):
            item = {
                "id": id,
                "source": source,
                "target": target,
                "candidates": [],
            }
            items.append(item)
        ds = CustomDataset(items)
        ds.logger.info(f"Dataset initialized from raw data; {len(ds.items)} items")
        return ds

    @classmethod
    def from_jsonl(cls, file:str):
        """
        load the dataset from a jsonl file
        """
        with open(file, 'r') as f:
            items = [json.loads(line) for line in f.readlines()]
        ds = CustomDataset(items, file)
        ds.logger.info(f"Dataset loaded from josnline file '{file}'")
        return ds

    def to_jsonl(self, file:str):
        """
        save the dataset to a jsonl file
        """
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        if file is not None:
            with open(file, 'w') as f:
                for item in self.items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _set_logger(self):
        """
        set the logger
        """
        self.logger = log.getLogger(__name__)
        self.logger.handlers = []
        self.logger.propagate = False
        self.logger.setLevel(log.INFO)
        format = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'logs', 'dataset.log')
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        fh = log.FileHandler(log_path)
        sh = log.StreamHandler()
        sh.setFormatter(format)
        fh.setFormatter(format)
        sh.setLevel(log.INFO)
        fh.setLevel(log.INFO)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def self_check(self):
        """
            check the data completeness
        """
        for item in self.items:
            for key in ['id', 'source', 'target', 'candidates']:
                assert key in item, f"Item must contain key '{key}':\n{item}"
            for cand in item['candidates']:
                assert set(cand) == set(['model', 'generation_method', 'text', 'scores']), f"Item must contain key '{key}':\n{item}"

        if self.file is not None:
            self.logger.info(f"Dataset loaded from file '{self.file}'")
        self.logger.info("Prepared metrics: " + str(self.prepared_metrics))
        candidate_counts = self.candidate_counts
        if len(candidate_counts) > 0:
            self.logger.info("Candidates models and generations:")
            for key, value in candidate_counts.items():
                self.logger.info(f"\t{key}: {value}")
            self.logger.info(f"Total candidates per source: {sum(candidate_counts.values())}")
        else:
            self.logger.info("No candidates added.")
        self.logger.info("Dataset size: {}".format(len(self.items)))

    def __getitem__(self, index):
        """
        get the item at index
        """
        return self.items[index]

    def __len__(self):
        """
        get the length of the dataset
        """
        return len(self.items)

    def __iter__(self):
        """
        iterate over the dataset
        """
        return iter(self.items)

    @property
    def prepared_metrics(self) -> Set[str]:
        """
        Get the list of metrics, that already computed for each item

        Returns:
            the set of metrics
        """
        computed_metrics = None
        for item in self.items:
            for cand in item['candidates']:
                if computed_metrics is None:
                    computed_metrics = set(cand['scores'].keys())
                else:
                    computed_metrics = computed_metrics.intersection(set(cand['scores'].keys()))
        return computed_metrics

    @property
    def candidate_counts(self) -> List[tuple]:
        """
        Get the list of candidate counts, that already computed for each item

        Returns:
            the Counter of candidates
        """
        counter = Counter()
        if len(self.items) == 0:
            return []
        for cand in self.items[0]['candidates']:
            counter[(cand['model'], cand['generation_method'])] += 1
        return counter


    def analyze_oracle(self) -> pt.PrettyTable:
        """
        Print the report
        k_best:
            report each of the performance selecting 1st, 2nd, ..., k_best hypotheses
            If None, print all the ranks
        """
        metrics = list(self.prepared_metrics)
        metrics.sort()

        table = pt.PrettyTable()
        fields = ['models', 'generation method']
        fields.extend([f"{metric}: top beam (min - mean - max)" for metric in metrics])
        table.field_names = fields
        table.align = 'l'
        # add original sources performances to the table
        rows = []
        types = set([(candidate['model'], candidate['generation_method']) for candidate in self.items[0]['candidates']])

        # 1. add oracle performances to the table for each model and generation method
        for model, generation_method in types:
            top_beam_scores = {metric: [] for metric in metrics}
            mean_scores = {metric: [] for metric in metrics}
            max_scores = {metric: [] for metric in metrics}
            min_scores = {metric: [] for metric in metrics}
            for item in self.items:
                candidates = [cand for cand in item['candidates'] if cand['model'] == model and cand['generation_method'] == generation_method]
                for metric in metrics:
                    top_beam_scores[metric].append(candidates[0]['scores'][metric])
                    mean_score = np.mean([cand['scores'][metric] for cand in candidates])
                    max_score = max([cand['scores'][metric] for cand in candidates])
                    min_score = min([cand['scores'][metric] for cand in candidates])
                    mean_scores[metric].append(mean_score)
                    max_scores[metric].append(max_score)
                    min_scores[metric].append(min_score)
            row = [model, generation_method]
            for metric in metrics:
                top_beam_score = np.mean(top_beam_scores[metric])
                mean_score = np.mean(mean_scores[metric])
                max_score = np.mean(max_scores[metric])
                min_score = np.mean(min_scores[metric])
                row.append(f"{top_beam_score:.3f} ({min_score:.3f} - {mean_score:.3f} - {max_score:.3f})")
            rows.append(row)

        # 2. add oracle performances to the table for each model using all generation methods
        model_counts = Counter()
        for model in [t[0] for t in types]:
            model_counts[model] += 1
        for model in set([t[0] for t in types]):
            if model_counts[model] == 1:
                continue
            top_beam_scores = {metric: [] for metric in metrics}
            mean_scores = {metric: [] for metric in metrics}
            max_scores = {metric: [] for metric in metrics}
            min_scores = {metric: [] for metric in metrics}
            for item in self.items:
                candidates = [cand for cand in item['candidates'] if cand['model'] == model]
                for metric in metrics:
                    top_beam_scores[metric].append(candidates[0]['scores'][metric])
                    mean_score = np.mean([cand['scores'][metric] for cand in candidates])
                    max_score = max([cand['scores'][metric] for cand in candidates])
                    min_score = min([cand['scores'][metric] for cand in candidates])
                    mean_scores[metric].append(mean_score)
                    max_scores[metric].append(max_score)
                    min_scores[metric].append(min_score)
            row = [model, "all"]
            for metric in metrics:
                top_beam_score = np.mean(top_beam_scores[metric])
                mean_score = np.mean(mean_scores[metric])
                max_score = np.mean(max_scores[metric])
                min_score = np.mean(min_scores[metric])
                row.append(f"{top_beam_score:.3f} ({min_score:.3f} - {mean_score:.3f} - {max_score:.3f})")
            rows.append(row)

        rows.sort(key=lambda x: x[0])
        table.add_rows(rows)
        self.logger.info(f"Oracle Analysis Resutls:\n{table}")
        return table
