from Models.MVG.gaussian_utils import *
import scipy


class LogGaussianClassifier:
    def __init__(self):
        
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.LTE = 0
        self.eff_prior = 0
        self.scores = 0
        self.logSPost = []
        self.name = "MVG"
        
        
    def train(self,DTR,LTR,DTE, LTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.eff_prior = eff_prior
        
        
        S = []
    
        sec_prior = 1 - self.eff_prior
        
        for i in range(self.LTR.max()+1):
            D_c = self.DTR[:,self.LTR == i] 
            mu,C = mean_and_covariance(D_c)
            f_conditional = logpdf_GAU_ND_fast(self.DTE, mu, C)
            S.append(mrow(f_conditional))
        S = numpy.vstack(S)
    
        prior = numpy.ones(S.shape) * [[self.eff_prior], [sec_prior]]
    
        logSJoint = S + numpy.log(prior)
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        self.logSPost = logSPost
        
    def compute_scores(self):
        sec_prior = 1- self.eff_prior
        llr = self.logSPost[1,:] - self.logSPost[0,:] - numpy.log(self.eff_prior/sec_prior)
        self.scores = llr
       
class NaiveBayesGaussianClassifier:
    def __init__(self):
        
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.LTE = 0
        self.eff_prior = 0
        self.scores = 0
        self.logSPost = []
        self.name = "NB"
        
        
    def train(self,DTR,LTR,DTE, LTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.eff_prior = eff_prior
        
        
        S = []
        sec_prior = 1 - self.eff_prior
    
        
    
        for i in range(self.LTR.max()+1):
            D_c = self.DTR[:,self.LTR == i] 
            mu,C = mean_and_covariance(D_c)
            identity = numpy.identity(C.shape[0])
            C = C*identity
            f_conditional = logpdf_GAU_ND_fast(self.DTE, mu, C)
            S.append(mrow(f_conditional))
        S = numpy.vstack(S)
    
        prior = numpy.ones(S.shape) * [[self.eff_prior], [sec_prior]]
    
   
        logSJoint = S + numpy.log(prior)
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        self.logSPost = logSPost
        
    def compute_scores(self):
        sec_prior = 1- self.eff_prior
        llr = self.logSPost[1,:] - self.logSPost[0,:] - numpy.log(self.eff_prior/sec_prior)
        self.scores = llr

class TiedGaussianClassifier:
    def __init__(self):
        
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.LTE = 0
        self.eff_prior = 0
        self.scores = 0
        self.logSPost = 0
        self.name = "TMVG"
        
        
    def train(self,DTR,LTR,DTE,LTE,eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.eff_prior = eff_prior
        
        
        sec_prior = 1 - self.eff_prior
    # Calculate the Tied Covariance Matrix
        C_star = 0 
        N = self.DTR.shape[1]
        for i in range(self.LTR.max()+1): 
            D_c = self.DTR[:, self.LTR == i] 
            nc = D_c.shape[1] 
            C_star = C_star + nc*mean_and_covariance(D_c)[1] 
    
        C_star = C_star / N
      
    # Apply Gaussian Classifier
        S = []
    
        for i in range(self.LTR.max()+1):
            D_c = self.DTR[:,self.LTR == i] 
            mu = mean_and_covariance(D_c)[0]
            f_conditional = logpdf_GAU_ND_fast(self.DTE, mu, C_star)
            S.append(mrow(f_conditional))
        S = numpy.vstack(S)
        
        prior = numpy.ones(S.shape) * [[self.eff_prior], [sec_prior]]
    
        logSJoint = S + numpy.log(prior)
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        self.logSPost = logSPost

        
    def compute_scores(self):
        sec_prior = 1- self.eff_prior
        llr = self.logSPost[1,:] - self.logSPost[0,:] - numpy.log(self.eff_prior/sec_prior)
        self.scores = llr
    
class TiedNaiveBayesGaussianClassifier:
    def __init__(self):
        
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.LTE = 0
        self.eff_prior = 0
        self.scores = 0
        self.logSPost = 0
        self.name = "TNB"
        
        
    def train(self,DTR,LTR,DTE,LTE,eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.eff_prior = eff_prior
        
        
        sec_prior = 1 - self.eff_prior
    # Calculate the Tied Covariance Matrix
        C_star = 0 
        N = self.DTR.shape[1]
        for i in range(self.LTR.max()+1): 
            D_c = self.DTR[:, self.LTR == i] 
            nc = D_c.shape[1] 
            C_star = C_star + nc*mean_and_covariance(D_c)[1] 
    
        C_star = C_star / N
    
    # Diagonalize the covariance matrix
        identity = numpy.identity(C_star.shape[0])
        C_star = C_star*identity
    
    # Apply Gaussian Classifier
        S = []
    
        for i in range(self.LTR.max()+1):
            D_c = self.DTR[:,self.LTR == i] 
            mu = mean_and_covariance(D_c)[0]
            f_conditional = logpdf_GAU_ND_fast(self.DTE, mu, C_star)
            S.append(mrow(f_conditional))
        S = numpy.vstack(S)  
    
        prior = numpy.ones(S.shape) * [[self.eff_prior], [sec_prior]]
    
        logSJoint = S + numpy.log(prior)
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        self.logSPost = logSPost
        
    def compute_scores(self):
        sec_prior = 1- self.eff_prior
        llr = self.logSPost[1,:] - self.logSPost[0,:] - numpy.log(self.eff_prior/sec_prior)
        self.scores = llr