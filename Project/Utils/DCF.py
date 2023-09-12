import numpy


def optimalBinaryBayesDecision(llr,pi,Cf_n,Cf_p):
    
    t = -numpy.log(pi*Cf_n/((1-pi)*Cf_p)) # threshold
    Predicted = (llr > t).astype(int)
    return Predicted

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
