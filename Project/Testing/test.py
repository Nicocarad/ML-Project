from Training.GMM.gmm import *
from Training.Logistic_Regression.logistic_regression import *
from Training.SVM.svm import *
from Preprocessing.Znorm import *
from Preprocessing.PCA import *
from Utils.DCF import *



def GMM_test_best(DTR, DTE, LTR, LTE):
    DTR, DTE = PCA(DTR, 11, DTE)
    iterations = 2
    gmm = GMM(iterations)
    gmm.train(DTR, LTR, DTE, LTE, None)
    gmm.compute_scores()
    scores = gmm.scores

    return scores, LTE




def prova(DTR,DTE,LTR,LTE):
    
    
  
        svm = RadialKernelBasedSvm(1,10,0.001)

        svm.train(DTR,LTR,DTE,LTE,0.5)
        svm.compute_scores()
        scores = svm.scores
        
        res_2 = min_DCF(0.5, 1, 1, LTE, scores)
        print("mindcf: ", res_2)
        





def SVM_test_best(DTR, DTE, LTR, LTE):
    DTR = znorm(DTR)
    DTE = znorm(DTE)
    lbd = 0.1
    C = 5
    pi_T = 0.5

    svm = RadialKernelBasedSvm(1, C, lbd)

    svm.train(DTR, LTR, DTE, LTE, pi_T)
    svm.compute_scores()
    scores = svm.scores

    return scores, LTE
