import torch
import numpy as np
import math
import random

def CELoss(logits, targets):
    """Compute cross-entropy loss for a batch of logits and targets.
    Subtract maximum logit value to prevent overflow."""
    # we will take the softmax over the last dimension (vocab_size)
    # we will return the loss averaged over the batch/seq_len dimensions

    # get maximum along each softmax dimension
    offset = torch.max(logits, dim = -1, keepdim = True)[0]

    # subtract largest element
    logits_offset = logits - offset

    # apply softmax
    xs = torch.gather(logits_offset, dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)
    xs -= torch.log(torch.sum(torch.exp(logits_offset), dim = -1, keepdim = False))

    # per token loss
    return -1 * torch.mean(xs)