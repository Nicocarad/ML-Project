
from Training.GMM.gmm_prova import *
from Utils.PCA import *
from Utils.DCF import *
import numpy as np




def GMM_test_best(DTR,DTE,LTR,LTE):
    
    DTR,DTE = PCA(DTR,12,DTE)
    
    # DTE = PCA(DTE,12)
    # pca = Pca(12)
    # DTR,DTE = pca.process(DTR,DTE)
    prova = np.load("Testing\GMMDiagonal_PCA_M_12_components_2_alpha_0.1_psi_0.01.npy")
    print(prova)
    iterations = 1  
    gmm = GMMDiagonal(iterations)
    gmm.train(DTR,LTR,DTE,LTE,None)
    gmm.compute_scores()
    scores = gmm.scores
    print(scores)
    res = min_DCF(0.5,1,1,LTE,scores)
    print("min_dcf", res)
    
    
    