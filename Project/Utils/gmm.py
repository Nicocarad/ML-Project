import numpy
import scipy
from Training.GMM.gmm_utils import *


def logpdf_gmm(X, gmm):
    s = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(X.shape[1]):
        for (idx, component) in enumerate(gmm):
            s[idx, i] = logpdf_GAU_ND_fast(X[:, i:i+1], component[1], component[2]) + numpy.log(component[0])
    return scipy.special.logsumexp(s, axis=0)



def em_algorithm(X, gmm, psi, fun_mod=None):
    ll_new = None
    ll_old = None
    while ll_old is None or ll_new - ll_old > 1e-6:
        
        num_components = len(gmm)
        ll_old = ll_new
        logS = numpy.zeros((num_components, X.shape[1]))
        
        # START E-STEP
        for g in range(num_components):
            logS[g, :] = logpdf_GAU_ND_fast(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        logSMarginal = scipy.special.logsumexp(logS, axis=0) #compute marginal densities
        ll_new = logSMarginal.sum() / X.shape[1]
        SPost = numpy.exp(logS - logSMarginal)
        
        # END E-STEP
        
        # START M-STEP
        gmm_new = []
        Z = numpy.zeros(num_components)
        
        for g in range(num_components):
            gamma = SPost[g, :]
            
            #update model parameters
            zero_order = gamma.sum()
            first_order = (mrow(gamma) * X).sum(1)
            second_order = numpy.dot(X, (mrow(gamma) * X).T)
            
            Z[g] = zero_order
            
            #new parameters
            mu = mcol(first_order / zero_order)
            sigma = second_order / zero_order - numpy.dot(mu, mu.T)
            w = zero_order / X.shape[1]
            
            
            gmm_new.append((w, mu, sigma))
        # END M-STEP

        if fun_mod is not None:
            gmm_new = fun_mod(gmm_new, Z, X.shape[1])

        # Constraining the eigenvalues
        for i in range(num_components):
            covNew = gmm_new[i][2]
            U, s, _ = numpy.linalg.svd(covNew)
            s[s < psi] = psi
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], numpy.dot(U, mcol(s) * U.T))
        gmm = gmm_new
          
        
    return gmm



def lbg_algorithm(iterations, X, start_gmm, alpha, psi, fun_mod=None):

    if fun_mod is not None:
        start_gmm = fun_mod(start_gmm, [X.shape[1]], X.shape[1])
        
    for i in range(len(start_gmm)):
        covNew = start_gmm[i][2]
        U, s, _ = numpy.linalg.svd(covNew)
        s[s < psi] = psi
        start_gmm[i] = (start_gmm[i][0], start_gmm[i]
                        [1], numpy.dot(U, mcol(s)*U.T))
    start_gmm = em_algorithm(X, start_gmm, psi, fun_mod)

    for i in range(iterations):
        gmm_new = list()
        for g in start_gmm:
            new_w = g[0]/2
            
            sigma_g = g[2]
            U, s, _ = numpy.linalg.svd(sigma_g)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            
            gmm_new.append((new_w, g[1] + d, sigma_g))
            gmm_new.append((new_w, g[1] - d, sigma_g))
        start_gmm = em_algorithm(X, gmm_new, psi, fun_mod)
    return start_gmm



def compute_gmm_scores(D, L, gmm):
    scores = numpy.zeros((numpy.unique(L).size, D.shape[1]))
    for classes in range(numpy.unique(L).size):
        scores[classes, :] = numpy.exp(logpdf_gmm(D, gmm[classes]))
    llr = numpy.zeros(scores.shape[1])
    for i in range(scores.shape[1]):
        llr[i] = numpy.log(scores[1, i] / scores[0, i])
    return llr

class GMM:

    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.DTR = 0
        self.DTE = 0
        self.LTR = 0
        self.gmm = None
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.scores = 0

    def compute_scores(self):
        self.scores = compute_gmm_scores(self.DTE,  self.LTE, self.gmm)

    def train(self,DTR,LTR,DTE,eff_prior):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR
        gmm = list()
        for classes in range(numpy.unique(self.LTR).size):
            mu, cov = mean_and_covariance(self.DTR[:, self.LTR == classes])
            gmm.append(lbg_algorithm(self.iterations, self.DTR[:, self.LTR == classes], [[1, mu, cov]], 0.1, 0.01))
        self.gmm = gmm

    

    