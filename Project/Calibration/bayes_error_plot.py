
import numpy as np
import matplotlib.pyplot as plt
from Utils.DCF import *



def bayes_error_plot(llr, Label, name):
    
    
    effPriorLogOdds = np.linspace(-4, 4, 100)
    
    dcf = []
    mindcf = []
    
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        predicted_labels = optimalBinaryBayesDecision(llr, pi, 1, 1)
        conf_matrix = confusionMatrix(Label, predicted_labels)
        dcf.append(DCF(pi, 1, 1, conf_matrix, "normalized"))
        mindcf.append(min_DCF(pi, 1, 1, Label, llr))
        
    plt.plot(effPriorLogOdds, dcf, label="DCF", color="r")
    plt.plot(effPriorLogOdds, mindcf, label="min DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF value')
    plt.title('Bayes error plot')
    plt.legend(loc="upper left")
    plt.savefig("Calibration/Plot/" + name + "_bayes_err_plot.pdf")
    plt.close()

    
    
    