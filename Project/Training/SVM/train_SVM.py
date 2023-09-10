from Utils.DCF import min_DCF
from Training.SVM.linear_svm import *
from Utils.Kfold import kfold
import matplotlib.pyplot as plt
from Utils.Znorm import *






def Lin_SVM(D,L):
    
    linear_svm = Linear_SVM(1,10)

    SPost_1, Label_1 = kfold(linear_svm, 5, D, L, 0.5)
    print("Before DCF", SPost_1.shape)
    res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
    print("min_DCF", res_1)
    
    
    
def SVM_RAW_znorm_01(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    value = [0.5]
    
    
    min_dcf_results_05 = []
    min_dcf_results_05_znorm = []
    

    for i, c in enumerate(C_values):
        svm = Linear_SVM(1,c)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.1, 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)
        print(i)
      
    D = znorm(D)  
    for i, c in enumerate(C_values):
        svm =  Linear_SVM(1,c)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.1, 1, 1, Label_2, SPost_2)
        min_dcf_results_05_znorm.append(res_2)
        print(i)
    
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    

    plt.plot(C_values, min_dcf_results_05, label="minDCF(\u03C0 = 0.1) RAW")
    plt.plot(C_values, min_dcf_results_05_znorm, label="minDCF(\u03C0 = 0.1) Z-norm")
    
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/Lin_SVM_RAW_Znorm_01.pdf")
    plt.close()
    
    
    
def SVM_RAW_znorm_09(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    value = [0.5]
    
    
    min_dcf_results_05 = []
    min_dcf_results_05_znorm = []
    

    for i, c in enumerate(C_values):
        svm = Linear_SVM(1,c)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.9, 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)
        print(i)
      
    D = znorm(D)  
    for i, c in enumerate(C_values):
        svm =  Linear_SVM(1,c)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.9, 1, 1, Label_2, SPost_2)
        min_dcf_results_05_znorm.append(res_2)
        print(i)
    
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    

    plt.plot(C_values, min_dcf_results_05, label="minDCF(\u03C0 = 0.9) RAW")
    plt.plot(C_values, min_dcf_results_05_znorm, label="minDCF(\u03C0 = 0.9) Z-norm")
    
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/Lin_SVM_RAW_Znorm_09.pdf")
    plt.close()