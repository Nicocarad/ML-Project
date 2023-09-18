import numpy as np
from Utils.utils import *


def transform_dataset(DTR, DTE):
    new_DTR = np.apply_along_axis(vec_xxT, 0, DTR)
    new_DTR = np.vstack([new_DTR, DTR])

    new_DTE = np.apply_along_axis(vec_xxT, 0, DTE)
    new_DTE = np.vstack([new_DTE, DTE])

    return new_DTR, new_DTE


def vec_xxT(x):
    x = mcol(x)
    return np.dot(x, x.T).reshape(x.size**2)
