import numpy
import scipy
import Library.functions as lib



def mcol(v):
    return v.reshape((v.size, 1)) # takes a row vector and emit a column vector

def load(fname): 
    DList = []
    labelsList = []
    hLabels = {
        '0': 0,
        '1': 1
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12] # list of string ( the first 4 string)
                attrs = mcol(numpy.array([float(i) for i in attrs])) # list compr
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except: #if the file contains an empty line at the end
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def mean_and_covariance(data_matrix):
    N = data_matrix.shape[1]
    mu = lib.vcol(data_matrix.mean(1)) 
    DC = data_matrix - mu 
    C = numpy.dot(DC, DC.T)/N
    
    return mu, C






   
def LogGaussianClassifier(DTR,LTR,DTE,LTE,eff_prior):
    
    S = []
    
    sec_prior = 1 - eff_prior
    for i in range(LTR.max()+1):
        D_c = DTR[:,LTR == i] 
        mu,C = mean_and_covariance(D_c)
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(lib.vrow(f_conditional))
    S = numpy.vstack(S)
    
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    
    logSJoint = S + numpy.log(prior)
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
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
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C)
        S.append(lib.vrow(f_conditional))
    S = numpy.vstack(S)
    
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    print(S.shape)
   
    logSJoint = S + numpy.log(prior)
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
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
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C_star)
        S.append(lib.vrow(f_conditional))
    S = numpy.vstack(S)
        
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    
    logSJoint = S + numpy.log(prior)
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
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
        f_conditional = lib.logpdf_GAU_ND_fast(DTE, mu, C_star)
        S.append(lib.vrow(f_conditional))
    S = numpy.vstack(S)  
    
    prior = numpy.ones(S.shape) * [[eff_prior], [sec_prior]]
    
    logSJoint = S + numpy.log(prior)
    logSMarginal = lib.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    llr = logSPost[1,:] - logSPost[0,:] - numpy.log(eff_prior/sec_prior)
    
    
    return  llr








def kfold(model,k,D,L,eff_prior=None,seed=0):
    

    SPost_partial = []
    folds = []
     
     # Create a list with indices of the Label vector in a random order
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    
    Label = L[idx] # randomize the vector of Real_labels in the same way

    
    fold_size = D.shape[1] // k

    # Divide indices in k-folds
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds.append(idx[start:end])
    

    # If the number of samples is not divisible by K, add the leavings samples in the last fold
    if D.shape[1] % k != 0:
        folds[-1] = numpy.concatenate((folds[-1], idx[k * fold_size:]))
        
    # Perform Cross validation
    for i in range(k):
        # Choose the i-th fold as validation fold
        validation_indices = folds[i]
        DTE = D[:,validation_indices] 
        LTE = L[validation_indices]
        # Use the leaving folds as Training Set
        train_indices = numpy.concatenate([folds[j] for j in range(k) if j != i])
        DTR = D[:,train_indices] 
        LTR = L[train_indices]
        # Append in the list the Scores (posterior probabilities) for the samples of the training fold
        if eff_prior is None:
            
           Spost = model(DTR, LTR, DTE, LTE)
        else:
           Spost = model(DTR, LTR, DTE, LTE,eff_prior)
        
       
        SPost_partial.append(Spost)
       
          
    S = numpy.hstack(SPost_partial) 
      
    
    return S,Label

def confusionMatrix(Real,Predicted):
    
    K = Real.max()+1 # number of classes
    confMatrix = numpy.zeros((K,K), dtype=int)
    
    for i,j in zip(Predicted,Real):
        confMatrix[i, j] += 1
    
    return confMatrix

def DCF(pi,C_fn,C_fp,confMatrix,type):
    
    
    (TN, FN), (FP, TP) = confMatrix
    
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    
    DCFu = pi*C_fn*FNR + (1-pi)*C_fp*FPR
    
    if type == "un-normalized":
        return DCFu
    elif type == "normalized":
        Bdummy = min(pi * C_fn, (1 - pi) * C_fp)
        DCFn = DCFu/Bdummy
        return DCFn
    else:
        raise ValueError('type must be either "un-normalized" or "normalized"')

def min_DCF(pi, C_fn, C_fp,LTE,scores):
    
    t = numpy.concatenate([scores, [-numpy.inf, numpy.inf]])
    t.sort()

    result = []
    for i in range(len(t)):
        
        Predicted = (scores > t[i]).astype(int)
        conf_matrix = confusionMatrix(LTE, Predicted)
        result.append(DCF(pi,C_fn,C_fp,conf_matrix,"normalized"))
        
    return min(result)


if __name__ == '__main__':
    
    
    #DTR: Training data
    #LTR: Training labels
    #DTE: Evaluation data
    #LTE: Evaluation labels
    D,L = load("Train.txt")
    
    SPost,Label = kfold(Tied_NaiveBayes_GaussianClassifier,5,D,L,0.1)
    res = min_DCF(0.1,1,1,Label,SPost)
    print("Min DCF: ", res)
    

    
    
    
    
    
    
    
    
    
    