from Utils.DCF import min_DCF
from Training.SVM.svm import *
from Utils.Kfold import kfold
import matplotlib.pyplot as plt
from Utils.Znorm import *

 
    
def SVM_RAW_znorm_01(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    
    
    
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
    
    
    
    
def SVM_diff_priors(D, L):
    C = 10
    priors = [(0.5, 0.5), (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.1, 0.1), (0.1, 0.9), (0.9, 0.5), (0.9, 0.1), (0.9, 0.9)]

    for pi_T, pi in priors:
        svm = Linear_SVM(1,C)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")
        

def SVM_diff_priors_znorm(D, L):
    C = 10
    priors = [(0.5, 0.5), (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.1, 0.1), (0.1, 0.9), (0.9, 0.5), (0.9, 0.1), (0.9, 0.9)]
    D = znorm(D)
    for pi_T, pi in priors:
        svm = Linear_SVM(1,C)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF_znorm (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")
        
                
        
def Poly_SVM_RAW_znorm_05(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    
    
    
    min_dcf_results_05 = []
    min_dcf_results_05_znorm = []
    

    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1,c,2,1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)
        print(i)
      
    D = znorm(D)  
    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1,c,2,1)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_05_znorm.append(res_2)
        print(i)
    
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    
    plt.plot(C_values, min_dcf_results_05, label="minDCF(\u03C0 = 0.5) RAW")
    plt.plot(C_values, min_dcf_results_05_znorm, label="minDCF(\u03C0 = 0.5) Z-norm")
    
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/Poly_SVM_RAW_Znorm_05.pdf")
    plt.close()
    
    
    
def Poly_SVM_RAW_znorm_01(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    
    
    
    min_dcf_results_05 = []
    min_dcf_results_05_znorm = []
    

    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1,c,2,1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.1, 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)
        print(i)
      
    D = znorm(D)  
    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1,c,2,1)

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
    plt.savefig("Training/SVM/Plot/Poly_SVM_RAW_Znorm_01.pdf")
    plt.close()
    
    
def Poly_SVM_RAW_znorm_09(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    
    
    
    min_dcf_results_05 = []
    min_dcf_results_05_znorm = []
    

    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1,c,2,1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.9, 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)
        print(i)
      
    D = znorm(D)  
    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1,c,2,1)

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
    plt.savefig("Training/SVM/Plot/Poly_SVM_RAW_Znorm_09.pdf")
    plt.close()   
    
    
    
def test(D,L):
    
    
    svm = RadialKernelBasedSvm(1,0.00001,0.1)

    SPost_1, Label_1 = kfold(svm, 5, D, L, 0.5)
    res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
    print("min_dcf", res_1)
    
    
    
    
def RadKernBased_RAW(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    
    min_dcf_results_log1= []
    min_dcf_results_log2 = []
    min_dcf_results_log3 = []
    
    

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log1.append(res_1)
        print(i)
      
      
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.1)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log2.append(res_2)
        print(i)
        
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.01)

        SPost_3, Label_3 = kfold(svm, 5, D, L, prior)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        min_dcf_results_log3.append(res_3)
        print(i)
    
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    
    
    plt.plot(C_values, min_dcf_results_log1, label="logγ = 0")
    plt.plot(C_values, min_dcf_results_log2, label="logγ = -1")
    plt.plot(C_values, min_dcf_results_log3, label="logγ = -2")
    
    
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/RadKernBased_RAW.pdf")
    plt.close() 
    
def RadKernBased_znorm(D, L, prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)
    D = znorm(D)
    
    min_dcf_results_log1= []
    min_dcf_results_log2 = []
    min_dcf_results_log3 = []
    
    

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log1.append(res_1)
        print(i)
      
      
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.1)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log2.append(res_2)
        print(i)
        
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.01)

        SPost_3, Label_3 = kfold(svm, 5, D, L, prior)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        min_dcf_results_log3.append(res_3)
        print(i)
    
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    
    
    plt.plot(C_values, min_dcf_results_log1, label="logγ = 0")
    plt.plot(C_values, min_dcf_results_log2, label="logγ = -1")
    plt.plot(C_values, min_dcf_results_log3, label="logγ = -2")
    
    
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/RadKernBased_znorm.pdf")
    plt.close()



def test(D,L):
    D = znorm(D)
    svm = RadialKernelBasedSvm(1,10,0.01)
    SPost_3, Label_3 = kfold(svm, 5, D, L, 0.5)
    res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
    print("min_dcf",res_3)