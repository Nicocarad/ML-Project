import numpy


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