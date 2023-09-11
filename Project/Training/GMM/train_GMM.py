from Utils.Kfold import *
from Utils.DCF import min_DCF
from Training.GMM.gmm_prova import *







def test_gmm(D,L):
    
    gmm = GMM_class(2)
    SPost_3, Label_3 = kfold(gmm, 5, D, L, None)
    res_3 = min_DCF(0.1, 1, 1, Label_3, SPost_3)
    print("min_dcf",res_3)
    
    
    