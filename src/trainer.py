import transformers
import torch
import torch.nn as nn
import torch.optim
class DualFiDTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        (index, target_ids, target_mask, context_ids, context_masks, scores) = inputs

        # compute the generation loss
        generation_loss = model(
            input_ids=context_ids.cuda(),
            attention_mask=context_masks.cuda(),
            labels=target_ids.cuda()
        )[0]
        # compute the mutli-task auxiliary loss, for this batch
        aux_loss = model.compute_auxiliary_loss(scores)
        # compute the balanced total loss
        loss = generation_loss + aux_loss * 10
        # set the outputs
        outpus = {}


        return (loss, generation_loss, aux_loss)




