import numpy as np
import matplotlib.pyplot as plt
from Utils.Kfold import kfold
from Utils.DCF import min_DCF
from Training.Logistic_Regression.logistic_regression import Logistic_Regression
from Utils.PCA import PCA



def plot_RAW_results(min_dcf_01,min_dcf_05,min_dcf_09):
    
    lambda_values = np.logspace(-5, 2, num=31)
    plt.figure()
    plt.xlabel('\u03BB')
    plt.xscale('log')
    plt.ylabel("minDCF")
    plt.title("RAW")

    plt.plot(lambda_values, min_dcf_05, label="minDCF(\u03C0 = 0.5)")
    plt.plot(lambda_values, min_dcf_01, label="minDCF(\u03C0 = 0.1)")
    plt.plot(lambda_values, min_dcf_09, label="minDCF(\u03C0 = 0.9)")
    plt.xlim(lambda_values[0], lambda_values[-1])
    plt.legend()

   
    plt.savefig("Training/Logistic_Regression/Plot/LR_RAW.pdf")
    plt.close()




def LR_RAW(D, L):
    
    l_values = np.logspace(-5, 2, num=31)

    value = [0.5, 0.1, 0.9]

    min_dcf_results_1 = []
    min_dcf_results_2 = []
    min_dcf_results_3 = []

    for i, l in enumerate(l_values):
        regression = Logistic_Regression(l)

        SPost_1, Label_1 = kfold(regression, 5, D, L, value[0])
        res_1 = min_DCF(value[0], 1, 1, Label_1, SPost_1)
        min_dcf_results_1.append(res_1)

        SPost_2, Label_2 = kfold(regression, 5, D, L, value[1])
        res_2 = min_DCF(value[1], 1, 1, Label_2, SPost_2)
        min_dcf_results_2.append(res_2)

        SPost_3, Label_3 = kfold(regression, 5, D, L, value[2])
        res_3 = min_DCF(value[2], 1, 1, Label_3, SPost_3)
        min_dcf_results_3.append(res_3)

        print(i)
    plot_RAW_results(min_dcf_results_1,min_dcf_results_2,min_dcf_results_3)

    








    
    



    