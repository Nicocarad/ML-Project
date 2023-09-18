import numpy


def mcol(array):
    return array.reshape((array.size, 1))


def mrow(array):
    return array.reshape((1,array.size))

def mean_and_covariance(data_matrix):
    N = data_matrix.shape[1]
    mu = mcol(data_matrix.mean(1)) 
    DC = data_matrix - mu 
    C = numpy.dot(DC, DC.T)/N
    
    return mu, C


def logpdf_GAU_ND_fast(X, mu, C):
    
    X_c = X - mu
    M = X.shape[0]
    const = - 0.5 * M * numpy.log(2*numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (X_c*numpy.dot(L, X_c)).sum(0)
    
    return const - 0.5 * logdet - 0.5 *v 


def compute_C_star(DTR, LTR):
    C_star = 0
    N = DTR.shape[1]
    max_label = numpy.max(LTR)
    
    for i in range(max_label + 1):
        D_c = DTR[:, LTR == i]
        nc = D_c.shape[1]
        C_star += nc * mean_and_covariance(D_c)[1] 
    
    C_star /= N
    
    return C_star