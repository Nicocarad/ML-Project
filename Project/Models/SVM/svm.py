import numpy
from scipy.optimize import fmin_l_bfgs_b
from Models.SVM.svm_utils import *






class Linear_SVM:
    
    def __init__(self,K,C):
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.LTE = 0
        self.K = K
        self.C = C
        self.priors = 0
        self.w = None
        self.scores = 0
        
    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.priors = [eff_prior, 1 - eff_prior]
        
        D_hat = numpy.vstack([self.DTR, numpy.ones(self.DTR.shape[1]) * self.K])
        G_hat = numpy.dot(D_hat.T, D_hat)
        z = 2 * self.LTR - 1
        H_hat = numpy.outer(z, z) * G_hat
        obj = compute_lagrangian_wrapper(H_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            obj,
            numpy.zeros(self.LTR.size),
            bounds=weighted_bounds(self.C, self.LTR, self.priors),
            factr=1.0
        )
        self.w = numpy.dot(D_hat, mcol(alpha) * mcol(z))
        self.DTE = numpy.vstack([self.DTE, numpy.ones(self.DTE.shape[1]) * self.K])
        
        
    def compute_scores(self):
        self.scores = numpy.dot(self.w.T, self.DTE)
        self.scores = self.scores.reshape(-1)
        
 
 
 
 

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
        self.alpha = 0

   

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.priors = [eff_prior, 1 - eff_prior]

        z = self.LTR * 2 - 1
        k_dtr = ((numpy.dot(self.DTR.T, self.DTR) + self.c) ** self.d) + (self.K ** 2)
        H_hat = mcol(z) * mrow(z) * k_dtr
        dual_obj = compute_lagrangian_wrapper(H_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            dual_obj,
            numpy.zeros(self.DTR.shape[1]),
            bounds=weighted_bounds(self.C, self.LTR, self.priors),
            factr=1.0)
        self.alpha = alpha

    def compute_scores(self):
        z = self.LTR * 2 - 1
        self.scores = (mcol(self.alpha) * mcol(z) * ((self.DTR.T.dot(self.DTE) + self.c) ** self.d + self.K ** 2)).sum(0)





class RadialKernelBasedSvm:

    def __init__(self, K, C, gamma):
       
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.K = K
        self.C = C
        self.gamma = gamma
        self.alpha = None
        self.scores = 0

    

    def train(self, DTR, LTR, DTE, LTE,  eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.priors = [eff_prior, 1 - eff_prior]
        
        z = 2 * self.LTR - 1
        RBF_kern_DTR = numpy.zeros((DTR.shape[1], DTR.shape[1]))
        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                RBF_kern_DTR[i, j] = numpy.exp(-self.gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + self.K **2
                
        H_hat = mcol(z) * mrow(z) * RBF_kern_DTR
        dual_obj = compute_lagrangian_wrapper(H_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            dual_obj,
            numpy.zeros(self.DTR.shape[1]),
            bounds=weighted_bounds(self.C, self.LTR, self.priors),
            factr=1.0
        )
        self.alpha = alpha

    def compute_scores(self):
        z = 2 * self.LTR - 1
        RBF_kern_DTE = numpy.zeros((self.DTR.shape[1], self.DTE.shape[1]))
        for i in range(self.DTR.shape[1]):
            for j in range(self.DTE.shape[1]):
                RBF_kern_DTE[i, j] = numpy.exp(-self.gamma * (numpy.linalg.norm(self.DTR[:, i] - self.DTE[:, j]) ** 2)) + self.K * self.K

        self.scores = numpy.sum(numpy.dot(self.alpha * mrow(z), RBF_kern_DTE), axis=0)