
import numpy

def mcol(v):
    return v.reshape((v.size, 1))


def mrow(array):
    return array.reshape((1,array.size))


# def mean_and_covariance(data_matrix):
#     N = data_matrix.shape[1]
#     mu = mcol(data_matrix.mean(1)) 
#     DC = data_matrix - mu 
#     C = numpy.dot(DC, DC.T)/N
    
#     return mu, C