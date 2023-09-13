
from Training.GMM.gmm import *
from Utils.PCA import *
from Utils.DCF import *
import numpy as np




def GMM_test_best(DTR,DTE,LTR,LTE):
    
    DTR,DTE = PCA(DTR,11,DTE)
    iterations = 2  
    gmm = GMM(iterations)
    gmm.train(DTR,LTR,DTE,LTE,None)
    gmm.compute_scores()
    scores = gmm.scores
    print(scores)
    res = min_DCF(0.5,1,1,LTE,scores)
    print("min_dcf", res)
    
    
    