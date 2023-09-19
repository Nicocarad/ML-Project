from Utils.utils import mcol
import numpy
import scipy
import scipy.linalg


def Sw_c(D_c):
    Sw_c = 0
    nc = D_c.shape[1]
    mu_c = mcol(D_c.mean(1))
    DC = D_c - mu_c
    Sw_c = numpy.dot(DC, DC.T) / nc
    return Sw_c


def SbSw(matrix, label):
    Sb = 0
    Sw = 0
    mu = mcol(matrix.mean(1))
    N = matrix.shape[1]
    for i in range(label.max() + 1):
        D_c = matrix[:, label == i]
        nc = D_c.shape[1]
        mu_c = mcol(D_c.mean(1))
        Sb = Sb + nc * numpy.dot((mu_c - mu), (mu_c - mu).T)
        Sw = Sw + nc * Sw_c(D_c)
    Sb = Sb / N
    Sw = Sw / N

    return Sb, Sw


def LDA1(matrix, label, m):
    Sb, Sw = SbSw(matrix, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    return W
