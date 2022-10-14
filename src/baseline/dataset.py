import json
import numpy as np

import torch


class GenerationDataset(torch.utils.data.Dataset):
    """
        Dataset for generate candidates for given sources
    """

    def __init__(self, tokenizer, sources, targets, source_max_length, target_max_length):
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        source = self.sources[item]
        target = self.targets[item]

        source_inputs = self.tokenizer(source, return_tensors="pt", max_length=self.source_max_length, padding='max_length')
        source_inputs["input_ids"] = source_inputs["input_ids"][:, :self.source_max_length]
        source_inputs["attention_mask"] = source_inputs["attention_mask"][:, :self.source_max_length]

        target_inputs = self.tokenizer(target, return_tensors="pt", max_length=self.target_max_length, padding='max_length')
        target_inputs["input_ids"] = target_inputs["input_ids"][:, :self.target_max_length]
        target_inputs["attention_mask"] = target_inputs["attention_mask"][:, :self.target_max_length]

        batch = {
            "source": source,
            "source_inputs": source_inputs,
            "target": target,
            "target_inputs": target_inputs,
        }

        return batch

