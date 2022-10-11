# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Some of the codes in this file are inherited from facebook FiD
# https://github.com/facebookresearch/FiD
# We thank the authors for their great work.

import transformers
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from src.wrapper import (
    EncoderWrapper,
    DualEncoderWrapper,
    DualDecoderWrapper,
)
class DualFiDBART(transformers.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.n_tasks = config.n_tasks
        self.wrap_model()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the BART forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.model.encoder.n_ctx = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.model.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_model(self):
        """
        Wrap the model's encoder, decoder with EncoderWrapper, DecoderWrapper.
        """
        encoder1 = self.model.encoder
        encoder2 = copy.deepcopy(encoder1)
        encoder2.embed_tokens = encoder1.embed_tokens # share the embedding
        self.model.encoder = DualEncoderWrapper(encoder1, encoder2, self.model.shared.padding_idx, self.n_tasks, self.config.d_model)
        self.model.decoder = DualDecoderWrapper(self.model.decoder)

    def unwrap_model(self):
        """
        Unwrap the model's encoder, decoder
        """
        self.model.encoder = self.model.encoder.encoder1
        self.model.decoder = self.model.decoder.decoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def compute_auxiliary_loss(self, scores):
        """
        Compute the auxiliary loss
        """
        sim, aux_loss = self.encoder.get_multi_task_layer_output()
        # compute contrastive loss
        if aux_loss is not None:
            loss = torch.tensor(aux_loss).to(sim.device)
        else:
            loss = torch.tensor(0.0).to(sim.device)
        e_sim = torch.exp(sim)
        print(e_sim.shape)
        labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(sim.device)
        e_sim_sum = torch.sum(e_sim, dim=1)
        # select a positive sample for each task and compute the loss
        e_sim_pos_sum = torch.sum(e_sim.unsqueeze(-1).expand(-1, -1, labels.size(-1)) * labels, dim=(1,2))
        loss += torch.mean(-torch.log(e_sim_pos_sum / e_sim_sum))
        return sim, loss

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    @property
    def shared(self):
        return self.model.shared

class DualFiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.n_tasks = config.n_tasks
        self.wrap_model()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_ctx = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, **kwargs):
        self.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            **kwargs
        )

    def wrap_model(self):
        """
        Wrap the model's encoder, decoder with EncoderWrapper, DecoderWrapper.
        """
        encoder1 = self.encoder
        encoder2 = copy.deepcopy(encoder1)
        encoder2.embed_tokens = encoder1.embed_tokens # share the embedding
        self.encoder = DualEncoderWrapper(encoder1, encoder2, self.config.pad_token_id, self.n_tasks, self.config.d_model)
        self.decoder = DualDecoderWrapper(self.decoder)

    def unwrap_model(self):
        """
        Unwrap the model's encoder, decoder
        """
        self.encoder = self.encoder.encoder1
        self.decoder = self.decoder.decoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def compute_auxiliary_loss(self, scores):
        """
        Compute the auxiliary loss
        Args:
            scores: (batch_size, n_candidates, n_tasks)
        """
        sim, aux_loss = self.encoder.get_multi_task_layer_output()
        # compute contrastive loss
        if aux_loss is not None:
            loss = torch.tensor(aux_loss).to(sim.device)
        else:
            loss = torch.tensor(0.0).to(sim.device)
        e_sim = torch.exp(sim)
        print(e_sim.shape)
        labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).to(sim.device)
        e_sim_sum = torch.sum(e_sim, dim=1)
        # select a positive sample for each task and compute the loss
        e_sim_pos_sum = torch.sum(e_sim.unsqueeze(-1).expand(-1, -1, labels.size(-1)) * labels, dim=(1,2))
        loss += torch.mean(-torch.log(e_sim_pos_sum / e_sim_sum))
        return sim, loss

class FiDBART(transformers.BartForConditionalGeneration):
    def __init__(self, config, device='cpu'):
        super().__init__(config)
        self.device = device
        self.wrap_encoder()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the BART forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.model.encoder.n_ctx = input_ids.size(1) - 1
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.model.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_encoder(self):
        """
        Wrap BART encoder to obtain a Fusion-in-Decoder model.
        """
        self.model.encoder = EncoderWrapper(self.model.encoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load BART weights.
        """
        self.model.encoder = self.model.encoder.encoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config, device='cpu'):
        super().__init__(config)
        self.device = device
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_ctx = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()









