import numpy as np
import scipy.optimize as opt


import numpy;


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


def kfold(model, k, D, L, eff_prior=None, seed=4):
    SPost_partial = []
    folds = []

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    
    Label = L[idx]

    fold_size = D.shape[1] // k

    # Divide indices into k-folds
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds.append(idx[start:end])

    # If the number of samples is not divisible by K, add the remaining samples to the last fold
    if D.shape[1] % k != 0:
        folds[-1] = np.concatenate((folds[-1], idx[k * fold_size:]))

    # Perform Cross validation
    for i in range(k):
        # Choose the i-th fold as the validation fold
        validation_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])

        model.train(D[:, train_indices], L[train_indices], D[:, validation_indices], eff_prior)
        model.compute_scores()
        scores = model.scores

        SPost_partial.append(scores)

    S = np.hstack(SPost_partial)

    return S, Label












def logreg_obj_wrap(model):
    
    z = 2*model.LTR -1 
        
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        s = 0
        priors = [model.eff_prior, 1- model.eff_prior]
        const = (model.l / 2) * (np.dot(w, w.T))
        for i in range(np.unique(model.LTR).size):
            const_class = (priors[i] / model.LTR[model.LTR == i].size)
            s += const_class * np.logaddexp(0, -z[model.LTR == i] * (np.dot(w.T, model.DTR[:, model.LTR == i]) + b)).sum()

        return const + s
    
    return logreg_obj
        
    
        

      
  
  
  
class Logistic_Regression():
    
    def __init__(self,l):
        
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.l = l
        self.eff_prior = 0
        self.w = None
        self.b = None
        self.scores = 0
  
    def train(self,DTR,LTR,DTE,eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.eff_prior = eff_prior
        logRegObj = logreg_obj_wrap(self)
        x,f,_ = opt.fmin_l_bfgs_b(logRegObj,np.zeros(self.DTR.shape[0] + 1),approx_grad = True,factr=1)
        
        self.w,self.b = x[0:-1], x[-1]
  
    
    def compute_scores(self):
        self.scores = np.dot(self.w.T,self.DTE) + self.b
        
  
  
def confusionMatrix(Real,Predicted):
    K = Real.max() + 1 # number of classes
    
    confMatrix = numpy.zeros((K,K), dtype=int)
    
    numpy.add.at(confMatrix, (Predicted, Real), 1)
    
    return confMatrix



def DCF(pi, C_fn, C_fp, confMatrix, type):
    TN, FN = confMatrix[0]
    FP, TP = confMatrix[1]
    
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    
    DCFu = pi * C_fn * FNR + (1 - pi) * C_fp * FPR
    
    if type == "un-normalized":
        return DCFu
    
    if type == "normalized":
        Bdummy = min(pi * C_fn, (1 - pi) * C_fp)
        DCFn = DCFu / Bdummy
        return DCFn
    
    raise ValueError('type must be either "un-normalized" or "normalized"')



def min_DCF(pi, C_fn, C_fp, LTE, scores):
    t = numpy.concatenate([scores, [-numpy.inf, numpy.inf]])
    t.sort()

    result = [DCF(pi, C_fn, C_fp, confusionMatrix(LTE, (scores > threshold).astype(int)), "normalized") for threshold in t]
        
    return min(result)
  
  
  
if __name__ == '__main__':
    
    D,L = load("Train.txt")
    l = 0
    regression = Logistic_Regression(l)
    
    SPost, Label = kfold(regression, 5, D, L, 0.5)
    res = min_DCF(0.5, 1, 1, Label, SPost)
    print("min_dcf", res)
  
  
  
  
  
  
  
  
  
  

    
       
       

    

    


    
