from Utils.DCF import min_DCF
from Training.SVM.linear_svm import *
from Utils.Kfold import kfold








def Lin_SVM(D,L):
    
    linear_svm = Linear_SVM(1,10)

    SPost_1, Label_1 = kfold(linear_svm, 5, D, L, 0.5)
    res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
    print("min_DCF", res_1)