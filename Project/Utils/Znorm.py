

def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))

def znorm(dataset):
    mean = mcol(dataset.mean(1))
    std = mcol(dataset.std(1))
    dataset = (dataset - mean) / std
    return dataset
