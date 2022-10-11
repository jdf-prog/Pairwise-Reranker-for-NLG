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
import copy

from src.wrapper import (
    EncoderWrapper,
    DualEncoderWrapper,
    DualDecoderWrapper,
    ModelMultitaskRegression,
    MoERegression
)
from transformers.models.bart.modeling_bart import shift_tokens_right
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

        # generate decoder input_ids from labels instead of input_ids
        if labels is None:
            decoder_input_ids
        else:
            decoder_input_ids = shift_tokens_right(
                labels if labels is not None else input_ids,
                self.config.pad_token_id,
                self.config.decoder_start_token_id)
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
        self.model.encoder = DualEncoderWrapper(encoder1, encoder2, self.model.shared.padding_idx)
        self.model.decoder = DualDecoderWrapper(self.model.decoder)
        self.multi_task_layer = ModelMultitaskRegression(
            self.n_tasks,
            self.model.config.d_model*2,
            self.model.config.d_model,
        )

    def unwrap_model(self):
        """
        Unwrap the model's encoder, decoder
        """
        self.model.encoder = self.model.encoder.encoder1
        self.model.decoder = self.model.decoder.decoder
        del self.multi_task_layer

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def compute_auxiliary_loss(self, scores):
        """
        Compute the auxiliary loss
        """
        source_cls_embed, candidate_cls_embed = self.model.encoder.get_cls_embed()
        bzs, n_candidates, d_model = candidate_cls_embed.size()
        inputs = torch.cat((
            source_cls_embed.unsqueeze(1).repeat(1, n_candidates, 1),
            candidate_cls_embed
        ), dim=-1)
        # compute the predictions scores
        x, aux_loss = self.multi_task_layer(inputs)
        # compute the loss
        if aux_loss is not None:
            loss = torch.tensor(aux_loss).to(x.device)
        else:
            loss = torch.tensor(0.0).to(x.device)
        x_ = x.reshape(-1, self.n_tasks)
        y_ = scores.reshape(-1, self.n_tasks).to(x_.device)
        for i in range(self.n_tasks):
            loss += torch.nn.functional.mse_loss(x_[:, i], y_[:, i], reduction='mean')
        loss /= self.n_tasks
        return x, loss


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
        self.encoder = DualEncoderWrapper(encoder1, encoder2, self.config.pad_token_id)
        self.decoder = DualDecoderWrapper(self.decoder)
        # self.multi_task_layer = ModelMultitaskRegression(
        #     self.n_tasks,
        #     self.config.d_model*2,
        #     self.config.d_model
        # )
        self.multi_task_layer = MoERegression(
            self.n_tasks,
            self.config.d_model*2,
            self.config.d_model
        )

    def unwrap_model(self):
        """
        Unwrap the model's encoder, decoder
        """
        self.encoder = self.encoder.encoder1
        self.decoder = self.decoder.decoder
        del self.multi_task_layer

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def compute_auxiliary_loss(self, scores):
        """
        Compute the auxiliary loss
        """
        source_cls_embed, candidate_cls_embed = self.encoder.get_cls_embed()
        bzs, n_candidates, d_model = candidate_cls_embed.size()
        inputs = torch.cat((
            source_cls_embed.unsqueeze(1).repeat(1, n_candidates, 1),
            candidate_cls_embed
        ), dim=-1)
        # compute the predictions scores
        x, aux_loss = self.multi_task_layer(inputs)
        # compute the loss
        if aux_loss is not None:
            loss = torch.tensor(aux_loss).to(x.device)
        else:
            loss = torch.tensor(0.0).to(x.device)
        x_ = x.reshape(-1, self.n_tasks)
        y_ = scores.reshape(-1, self.n_tasks).to(x_.device)
        for i in range(self.n_tasks):
            loss += torch.nn.functional.mse_loss(x_[:, i], y_[:, i], reduction='mean')
        loss /= self.n_tasks
        return x, loss

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

        # generate decoder input_ids from labels instead of input_ids
        decoder_input_ids = shift_tokens_right(
            labels,
            self.config.pad_token_id,
            self.config.decoder_start_token_id)
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
