import numpy
from Utils.utils import mcol



def PCA(data_matrix,m,DTE = None):
    
    N = data_matrix.shape[1] # number of samples
    mu = mcol(data_matrix.mean(1)) # data_matrix.mean(1) return a 1-D array so we must transform it into a column
    DC = data_matrix - mu  # performing broadcast, we are removing from all the data the mean
    C = numpy.dot(DC, DC.T)/N
    s, U = numpy.linalg.eigh(C) # compute the eigenvalues and eigenvectors
    P = U[:, ::-1][:, 0:m] # reverse the matrix in order to move leading eigenvectors in the first "m" column
    DP = numpy.dot(P.T, data_matrix) 
    if DTE is not None:
        DTE = numpy.dot(P.T, DTE)
    return DP, DTE