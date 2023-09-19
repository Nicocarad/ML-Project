import numpy
from Utils.utils import mcol


def PCA(data_matrix, m, DTE=None):
    N = data_matrix.shape[1]
    mu = mcol(data_matrix.mean(1))
    DC = data_matrix - mu
    C = numpy.dot(DC, DC.T) / N
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, data_matrix)
    if DTE is not None:
        DTE = numpy.dot(P.T, DTE)
    return DP, DTE
