import numpy as np
import random
import torch

def load_data(x: np.array, batch_size: int, seq_len: int, device: str, dtype: torch.dtype = torch.long, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch from the input array x, sampling sequences independently.
    x: input array of shape length x 1
    batch_size: number of sequences to sample
    seq_len: length of each sequence
    device: device to load the data onto
    """
    # assume x is of shape length x 1
    length = x.shape[0]
    start_indices = random.sample(range(0, length - seq_len), batch_size)
    # use the next line to debug on a single minibatch
    if not sample:
        start_indices = np.arange(0, batch_size)
    batch = np.zeros((batch_size, seq_len))
    targets = np.zeros((batch_size, seq_len))

    for i, p in enumerate(start_indices):
        batch[i, :] = x[p: p + seq_len]
        targets[i, :] = x[p + 1: p + 1 + seq_len]

    return torch.tensor(batch, device = device, dtype = dtype), torch.tensor(targets, device = device, dtype = dtype)

def save_checkpoint(model, optimizer, iteration, out):
    to_save = {"model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "iteration": iteration}
    torch.save(to_save, out)

def load_checkpoint(src, model, optimizer):
    loaded = torch.load(src)
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    return loaded["iteration"]