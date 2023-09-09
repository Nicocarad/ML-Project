

import numpy as np

def znorm(dataset):
    mu = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)

    normalized_dataset = (dataset - mu) / std

    return normalized_dataset

