from Training.Gaussian.gaussian_utils import *
import scipy


def LogGaussianClassifier(DTR,LTR,DTE,LTE,eff_prior):
    
    S = []
    
    sec_prior = 1 - eff_prior
    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        f_conditional = logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(mrow(f_conditional))
    S = numpy.vstack(S)
    
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    
    logSJoint = S + numpy.log(prior)
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    
    llr = logSPost[1,:] - logSPost[0,:] - numpy.log(eff_prior/sec_prior)
    return llr
       
def NaiveBayes_GaussianClassifier(DTR,LTR,DTE,LTE,eff_prior):
    
    
    sec_prior = 1 - eff_prior
    
    S = []
    
    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        identity = numpy.identity(C.shape[0])
        C = C*identity
        f_conditional = logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(mrow(f_conditional))
    S = numpy.vstack(S)
    
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    
   
    logSJoint = S + numpy.log(prior)
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    llr = logSPost[1,:] - logSPost[0,:] - numpy.log(eff_prior/sec_prior)

    return  llr

def TiedGaussianClassifier(DTR,LTR,DTE,LTE,eff_prior):
    
    sec_prior = 1 - eff_prior
    # Calculate the Tied Covariance Matrix
    C_star = 0 
    N = DTR.shape[1]
    for i in range(LTR.max()+1): 
        D_c = DTR[:, LTR == i] 
        nc = D_c.shape[1] 
        C_star = C_star + nc*mean_and_covariance(D_c)[1] 
    
    C_star = C_star / N
      
    # Apply Gaussian Classifier
    S = []
    
    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu = mean_and_covariance(D_c)[0]
        f_conditional = logpdf_GAU_ND_fast(DTE, mu, C_star)
        S.append(mrow(f_conditional))
    S = numpy.vstack(S)
        
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    
    logSJoint = S + numpy.log(prior)
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    llr = logSPost[1,:] - logSPost[0,:] - numpy.log(eff_prior/sec_prior)
    
    return llr
    
def Tied_NaiveBayes_GaussianClassifier(DTR,LTR,DTE,LTE,eff_prior):
    
    sec_prior = 1 - eff_prior
    # Calculate the Tied Covariance Matrix
    C_star = 0 
    N = DTR.shape[1]
    for i in range(LTR.max()+1): 
        D_c = DTR[:, LTR == i] 
        nc = D_c.shape[1] 
        C_star = C_star + nc*mean_and_covariance(D_c)[1] 
    
    C_star = C_star / N
    
    # Diagonalize the covariance matrix
    identity = numpy.identity(C_star.shape[0])
    C_star = C_star*identity
    
    # Apply Gaussian Classifier
    S = []
    
    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu = mean_and_covariance(D_c)[0]
        f_conditional = logpdf_GAU_ND_fast(DTE, mu, C_star)
        S.append(mrow(f_conditional))
    S = numpy.vstack(S)  
    
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    
    logSJoint = S + numpy.log(prior)
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    llr = logSPost[1,:] - logSPost[0,:] - numpy.log(eff_prior/sec_prior)
    
    
    return  llr