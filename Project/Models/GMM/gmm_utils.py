import numpy
import scipy
from Utils.utils import *


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

def logpdf_gmm(X, gmm):
    s = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(X.shape[1]):
        for (idx, component) in enumerate(gmm):
            s[idx, i] = logpdf_GAU_ND_fast(X[:, i:i+1], component[1], component[2]) + numpy.log(component[0])
    return scipy.special.logsumexp(s, axis=0)

def LBG(iterations, X, start_gmm, alpha, psi, covariance_func=None):

    if covariance_func is not None:
        start_gmm = covariance_func(start_gmm, [X.shape[1]], X.shape[1])
        
    for i in range(len(start_gmm)):
        covNew = start_gmm[i][2]
        U, s, _ = numpy.linalg.svd(covNew)
        s[s < psi] = psi
        start_gmm[i] = (start_gmm[i][0], start_gmm[i]
                        [1], numpy.dot(U, mcol(s)*U.T))
    start_gmm = EM(X, start_gmm, psi, covariance_func)

    for i in range(iterations):
        gmm_new = []
        for g in start_gmm:
            new_w = g[0]/2
            
            sigma_g = g[2]
            U, s, _ = numpy.linalg.svd(sigma_g)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            
            gmm_new.append((new_w, g[1] + d, sigma_g))
            gmm_new.append((new_w, g[1] - d, sigma_g))
        start_gmm = EM(X, gmm_new, psi, covariance_func)
    return start_gmm

def EM(X, gmm, psi, covariance_func=None):
    llr_1 = None
    llr_0 = None
    while llr_0 is None or llr_1 - llr_0 > 1e-6:
        
        num_components = len(gmm)
        llr_0 = llr_1
        logS = numpy.zeros((num_components, X.shape[1]))
        
        # START E-STEP
        for idx in range(num_components):
            logS[idx, :] = logpdf_GAU_ND_fast(X, gmm[idx][1], gmm[idx][2]) + numpy.log(gmm[idx][0])
        logSMarginal = scipy.special.logsumexp(logS, axis=0) #compute marginal densities
        llr_1 = numpy.mean(logSMarginal)
        SPost = numpy.exp(logS - logSMarginal)
        
        # END E-STEP
        
        # START M-STEP
        gmm_new = []
        Z_vec = numpy.zeros(num_components)
        
        for idx in range(num_components):
            gamma = SPost[idx, :]
            
            #update model parameters
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = numpy.dot(X, (mrow(gamma) * X).T)
            
            Z_vec[idx] = Z
            
            #new parameters
            mu = mcol(F / Z)
            sigma = S / Z - numpy.dot(mu, mu.T)
            w = Z / X.shape[1]
            
            
            gmm_new.append((w, mu, sigma))
        # END M-STEP

        if covariance_func is not None:
            gmm_new = covariance_func(gmm_new, Z_vec, X.shape[1])

        # Constraining the eigenvalues
        for i in range(num_components):
            covNew = gmm_new[i][2]
            U, s, _ = numpy.linalg.svd(covNew)
            s[s < psi] = psi
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], numpy.dot(U, mcol(s) * U.T))
        gmm = gmm_new
          
        
    return gmm

def gmm_scores(D, L, gmm):
    scores = numpy.zeros((numpy.unique(L).size, D.shape[1]))
    for classes in range(numpy.unique(L).size):
        scores[classes, :] = numpy.exp(logpdf_gmm(D, gmm[classes]))
    llr = numpy.zeros(scores.shape[1])
    for i in range(scores.shape[1]):
        llr[i] = numpy.log(scores[1, i] / scores[0, i])
    return llr

def tied_cov(gmm, z_vec, n):
        tied_sigma = numpy.zeros_like(gmm[0][2])
    
        for g in range(len(gmm)):
            tied_sigma += gmm[g][2] * z_vec[g]
    
        tied_sigma *= (1 / n)
    
        for i in range(len(gmm)):
            gmm[i] = (gmm[i][0], gmm[i][1], tied_sigma)
    
        return gmm  
    
def diagonal_cov(gmm, _z_vec, _n):
    gmm = [(g[0], g[1], numpy.diag(numpy.diag(g[2]))) for g in gmm]
    return gmm

def TiedDiagonal_cov(gmm, z_vec, n):
    tied_gmm = tied_cov(gmm, z_vec, n)
    tied_diagonal_gmm = diagonal_cov(tied_gmm, z_vec, n)
    return tied_diagonal_gmm