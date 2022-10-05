import transformers
import torch

class DualFiDTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ...
