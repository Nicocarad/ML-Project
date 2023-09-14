import matplotlib.pyplot as plt


from Training.SVM.svm import *
from Utils.DCF import *
from Utils.Kfold import *
from Preprocessing.Znorm import *


def RadKernBased_RAW_eval(DTR, LTR, DTE, LTE ,prior):
    
    
    C_values = numpy.logspace(-5, 5, num=11)

    
    min_dcf_results_log1= []
    min_dcf_results_log2 = []
    min_dcf_results_log3 = []
    min_dcf_results_log1_eval = []
    min_dcf_results_log2_eval = []
    min_dcf_results_log3_eval = []
    
    
    TOT  = 66
    j = 1

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.1)

        SPost_1, Label_1 = kfold(svm, 5, DTR, LTR, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log1.append(res_1)
        print("66/"+str(j))
        j = j+1
        
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.1)

        svm.train(DTR,LTR,DTE,LTE,0.5)
        svm.compute_scores()
        scores = svm.scores
        res_1 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log1_eval.append(res_1)
        print("66/"+str(j))
        j = j+1
      
      
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.01)

        SPost_2, Label_2 = kfold(svm, 5, DTR, LTR, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log2.append(res_2)
        print("66/"+str(j))
        j = j+1
        
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.01)

        svm.train(DTR,LTR,DTE,LTE,0.5)
        svm.compute_scores()
        scores = svm.scores
        
        res_2 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log2_eval.append(res_2)
        print("66/"+str(j))
        j = j+1
    
        
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.001)

        SPost_3, Label_3 = kfold(svm, 5, DTR, LTR, prior)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        min_dcf_results_log3.append(res_3)
        print("66/"+str(j))
        j = j+1
        
        
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1,c,0.001)

        svm.train(DTR,LTR,DTE,LTE,0.5)
        svm.compute_scores()
        scores = svm.scores
        res_3 = min_DCF(0.5, 1, 1,LTE,scores)
        min_dcf_results_log3_eval.append(res_3)
        print("66/"+str(j))
        j = j+1
    
    plt.figure()
    plt.xlabel('C')
    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.title("RKB-SVM")
    
    
    plt.plot(C_values, min_dcf_results_log1, label="logγ = -1 (VAL)" , color = "red")
    plt.plot(C_values, min_dcf_results_log2, label="logγ = -2 (VAL)", color = "blue")
    plt.plot(C_values, min_dcf_results_log3, label="logγ = -3 (VAL)", color = "green")
    plt.plot(C_values, min_dcf_results_log1_eval, label="logγ = -1 (EVAL)", color = "red", linestyle="dashed")
    plt.plot(C_values, min_dcf_results_log2_eval, label="logγ = -2 (EVAL)", color = "blue", linestyle="dashed")
    plt.plot(C_values, min_dcf_results_log3_eval, label="logγ = -3 (EVAL)", color = "green", linestyle="dashed")
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Testing/SVM/Plot/RadKernBased_RAW_eval.pdf")
    plt.close()