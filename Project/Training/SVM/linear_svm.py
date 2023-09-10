import numpy
from scipy.optimize import fmin_l_bfgs_b


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(array):
    return array.reshape((1,array.size))




class Linear_SVM:
    
    def __init__(self,K,C):
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.K = K
        self.C = C
        self.priors = 0
        self.w = None
        self.scores = 0
        
    def train(self, DTR, LTR, DTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.priors = [eff_prior, 1 - eff_prior]
        
        d_hat = numpy.vstack([self.DTR, numpy.ones(self.DTR.shape[1]) * self.K])
        g_hat = numpy.dot(d_hat.T, d_hat)
        z = 2 * self.LTR - 1
        h_hat = numpy.outer(z, z) * g_hat
        obj = obj_svm_wrapper(h_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            obj,
            numpy.zeros(self.LTR.size),
            bounds=compute_weights(self.C, self.LTR, self.priors),
            factr=1.0
        )
        self.w = numpy.dot(d_hat, mcol(alpha) * mcol(z))
        self.DTE = numpy.vstack([self.DTE, numpy.ones(self.DTE.shape[1]) * self.K])
        
        
    def compute_scores(self):
        self.scores = numpy.dot(self.w.T, self.DTE)
        self.scores = self.scores.reshape(-1)
        
 
 
 
 
        
def compute_weights(C, LTR, prior):
    bounds = numpy.zeros((LTR.shape[0]))
    empirical_pi_t = (LTR == 1).sum() / LTR.shape[0]
    bounds[LTR == 1] = C * prior[1] / empirical_pi_t
    bounds[LTR == 0] = C * prior[0] / (1 - empirical_pi_t)
    return list(zip(numpy.zeros(LTR.shape[0]), bounds))


def obj_svm_wrapper(H_hat):
    def obj_svm(alpha):
        alpha = mcol(alpha)
        gradient = mrow(H_hat.dot(alpha) - numpy.ones((alpha.shape[0], 1)))
        obj_l = 0.5 * alpha.T.dot(H_hat).dot(alpha) - alpha.T @ numpy.ones(alpha.shape[0])
        return obj_l, gradient

    return obj_svm



class PolynomialSvm:

    def __init__(self, K, C, d, c):
        
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.K = K
        self.C = C
        self.c = c
        self.d = d
        self.priors = 0
        self.scores = 0
        self.alpha = None

   

    def train(self, DTR, LTR, DTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.priors = [eff_prior, 1 - eff_prior]

        z = self.LTR * 2 - 1
        k_dtr = ((numpy.dot(self.DTR.T, self.DTR) + self.c) ** self.d) + (self.K ** 2)
        h_hat = mcol(z) * mrow(z) * k_dtr
        dual_obj = obj_svm_wrapper(h_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            dual_obj,
            numpy.zeros(self.DTR.shape[1]),
            bounds=compute_weights(self.C, self.LTR, self.priors),
            factr=1.0)
        self.alpha = alpha

    def compute_scores(self):
        z = self.LTR * 2 - 1
        self.scores = (
                mcol(self.alpha) * mcol(z) * ((self.DTR.T.dot(self.DTE) + self.c) ** self.d + self.K ** 2)).sum(0)
