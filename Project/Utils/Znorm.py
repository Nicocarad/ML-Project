

import numpy as np


def znorm(dataset):
    mean = compute_mean(dataset)
    std = compute_std(dataset)
    dataset = (dataset - mean) / std
    return dataset

def compute_mean(d: np.ndarray) -> np.ndarray:
    return mcol(d.mean(1))

def compute_std(d: np.ndarray) -> np.ndarray:
    return mcol(d.std(1))





def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))

