import numpy as np
import matplotlib.pyplot as plt
from Utils.Kfold import kfold
from Utils.DCF import min_DCF
from Training.Logistic_Regression.logistic_regression import Logistic_Regression
from Utils.PCA import PCA
from Utils.Znorm import *



def plot_RAW_results(min_dcf_05,min_dcf_01,min_dcf_09, name,title):
    
    lambda_values = np.logspace(-5, 2, num=41)
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.title(title)

    plt.plot(lambda_values, min_dcf_05, label="minDCF(\u03C0 = 0.5)")
    plt.plot(lambda_values, min_dcf_01, label="minDCF(\u03C0 = 0.1)")
    plt.plot(lambda_values, min_dcf_09, label="minDCF(\u03C0 = 0.9)")
    plt.xlim(lambda_values[0], lambda_values[-1])
    plt.legend()

   
    plt.savefig("Training/Logistic_Regression/Plot/" + name + ".pdf")
    plt.close()




def LR_RAW(D, L, prior):
    
    name = "LR_RAW"
    l_values = np.logspace(-5, 2, num=41)

    value = [0.5, 0.1, 0.9]
    
    
    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    for i, l in enumerate(l_values):
        regression = Logistic_Regression(l)

        SPost_1, Label_1 = kfold(regression, 5, D, L, prior)
        res_1 = min_DCF(value[0], 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)

        SPost_2, Label_2 = kfold(regression, 5, D, L, prior)
        res_2 = min_DCF(value[1], 1, 1, Label_2, SPost_2)
        min_dcf_results_01.append(res_2)

        SPost_3, Label_3 = kfold(regression, 5, D, L, prior)
        res_3 = min_DCF(value[2], 1, 1, Label_3, SPost_3)
        min_dcf_results_09.append(res_3)

        print(i)
    plot_RAW_results(min_dcf_results_05,min_dcf_results_01,min_dcf_results_09,name,"RAW")

    
def LR_RAW_Znorm(D, L, prior):
    
    D = znorm(D)
    name = "LR_RAW_znorm"
    l_values = np.logspace(-5, 2, num=41)

    value = [0.5, 0.1, 0.9]
    
    
    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    for i, l in enumerate(l_values):
        regression = Logistic_Regression(l)

        SPost_1, Label_1 = kfold(regression, 5, D, L, prior)
        res_1 = min_DCF(value[0], 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)

        SPost_2, Label_2 = kfold(regression, 5, D, L, prior)
        res_2 = min_DCF(value[1], 1, 1, Label_2, SPost_2)
        min_dcf_results_01.append(res_2)

        SPost_3, Label_3 = kfold(regression, 5, D, L, prior)
        res_3 = min_DCF(value[2], 1, 1, Label_3, SPost_3)
        min_dcf_results_09.append(res_3)

        print(i)
    plot_RAW_results(min_dcf_results_05,min_dcf_results_01,min_dcf_results_09,name,"Z-NORM")
    
    
    
    
def LR_PCA(D, L, prior):
    
    m = [11,10,9]
    
    name = "LR_PCA"
    l_values = np.logspace(-5, 2, num=41)

    value = [0.5]
    
    
    min_dcf_results_raw = []
    min_dcf_results_11 = []
    min_dcf_results_10 = []
    min_dcf_results_09 = []

    for l in l_values:
        regression = Logistic_Regression(l)
        SPost_1, Label_1 = kfold(regression, 5, D, L, prior)
        res_1 = min_DCF(value[0], 1, 1, Label_1, SPost_1)
        min_dcf_results_raw.append(res_1)
    print("END1")
    D_11 = PCA(D,11)
    for l in l_values:
        regression = Logistic_Regression(l)
        SPost_2, Label_2 = kfold(regression, 5, D_11, L, prior)
        res_2 = min_DCF(value[0], 1, 1, Label_2, SPost_2)
        min_dcf_results_11.append(res_2)
    print("END2")  
    D_10 = PCA(D,10)
    for l in l_values:
        regression = Logistic_Regression(l)
        SPost_3, Label_3 = kfold(regression, 5, D_10, L, prior)
        res_3 = min_DCF(value[0], 1, 1, Label_3, SPost_3)
        min_dcf_results_10.append(res_3)
    print("END3") 
    D_9 = PCA(D,9)
    for l in l_values:
        regression = Logistic_Regression(l)
        SPost_4, Label_4 = kfold(regression, 5, D_9, L, prior)
        res_4 = min_DCF(value[0], 1, 1, Label_4, SPost_4)
        min_dcf_results_09.append(res_4)

    
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.title("RAW + PCA")

    plt.plot(l_values, min_dcf_results_raw, label="minDCF(\u03C0 = 0.5) RAW")
    plt.plot(l_values, min_dcf_results_11, label="minDCF(\u03C0 = 0.5) PCA 11")
    plt.plot(l_values, min_dcf_results_10, label="minDCF(\u03C0 = 0.5) PCA 10")
    plt.plot(l_values, min_dcf_results_09, label="minDCF(\u03C0 = 0.5) PCA 9")
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend()

   
    plt.savefig("Training/Logistic_Regression/Plot/" + name + ".pdf")
    plt.close()
   

   





    
    

    
    



    