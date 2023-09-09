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
    m_values = [11, 10, 9]
    name = "LR_PCA"
    l_values = np.logspace(-5, 2, num=41)
    value = [0.5]

    min_dcf_results = [[] for _ in range(len(m_values))]

    for i, m in enumerate(m_values):
        D_pca = PCA(D, m)
        
        for l in l_values:
            regression = Logistic_Regression(l)
            SPost, Label = kfold(regression, 5, D_pca, L, prior)
            res = min_DCF(value[0], 1, 1, Label, SPost)
            min_dcf_results[i].append(res)

    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.title("RAW + PCA")

    m_labels = ['RAW', 'PCA 11', 'PCA 10', 'PCA 9']
    for i, results in enumerate(min_dcf_results):
        plt.plot(l_values, results, label=f"minDCF(\u03C0 = 0.5) {m_labels[i]}")

    plt.xlim(l_values[0], l_values[-1])
    plt.legend()

    plt.savefig("Training/Logistic_Regression/Plot/" + name + ".pdf")
    plt.close()
   

   





    
    

    
    



    