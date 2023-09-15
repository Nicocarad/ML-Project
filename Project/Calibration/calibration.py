
import numpy
from Utils.Kfold import *
from Models.LR.logistic_regression import *



def calibration(Scores, LTR, pi):
    prior_1 = pi
    prior_2 = 1 - pi
    numpy.random.seed(4)
    idx = numpy.random.permutation(Scores.size)
    LTR_shuff = LTR[idx]
    Scores_shuff = Scores[idx].reshape(1,Scores.size)
    logreg = Logistic_Regression(0)
    S,L = kfold(logreg,5,Scores_shuff,LTR_shuff,0.5)
    calibrated_score = S - numpy.log(prior_1 / prior_2)
    return calibrated_score, L
    
    
    
    
    
