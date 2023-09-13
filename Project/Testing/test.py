
from Training.GMM.gmm import *
from Utils.PCA import *
from Utils.DCF import *





def GMM_test_best(DTR,DTE,LTR,LTE):
    
    # DTR = PCA(DTR,11)
    # DTE = PCA(DTE,11)
    iterations = 2  
    gmm = GMM_Tied(iterations)
    gmm.train(DTR,LTR,DTE,LTE,None)
    gmm.compute_scores()
    scores = gmm.scores
    res = min_DCF(0.5,1,1,LTE,scores)
    print("min_dcf", res)
    