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
    Sb = 0  # initialize the between class cov. matrix
    Sw = 0  # initialize the within class cov. matrix
    mu = mcol(matrix.mean(1))  # dataset mean
    N = matrix.shape[1]
    for i in range(
        label.max() + 1
    ):  # in "label" there are only 0,1,2 element so the max will be 2
        D_c = matrix[
            :, label == i
        ]  # filter the matrix data according to the label (0,1,2)
        nc = D_c.shape[1]  # number of sample in class "c"
        mu_c = mcol(
            D_c.mean(1)
        )  # calc a column vector containing the mean of the attributes (sepal-length, petal-width ...) for one class at a time
        Sb = Sb + nc * numpy.dot((mu_c - mu), (mu_c - mu).T)
        Sw = Sw + nc * Sw_c(
            D_c
        )  # calculate the within covariance matrix as a weighted sum of the cov matrix of the classes
    Sb = Sb / N
    Sw = Sw / N

    return Sb, Sw


def LDA1(matrix, label, m):
    Sb, Sw = SbSw(matrix, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]  # reverse the eigenvectors and then retrive the first m
    return W