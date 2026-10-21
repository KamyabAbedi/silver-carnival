import numpy as np
import torch

def get_batch(data, batch_size, context_length, device):

    starts = np.random.randint(0, len(data) - context_length, size=batch_size)

    x = np.stack([data[i : i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )