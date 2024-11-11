import numpy as np
import torch


def generate_window(data, win_size):
    windows = []
    length = data.shape[0]
    for i in range(length):
        if i+win_size <= length:
            windows.append(data[i:i+win_size])
        else:
            break

    return torch.Tensor(np.array(windows))