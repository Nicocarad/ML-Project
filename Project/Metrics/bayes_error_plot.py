
import numpy as np
import matplotlib.pyplot as plt
from Metrics.DCF import *



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



def bayes_error_plot_multi(llr_LR, Label_LR, llr_SVM, Label_SVM, llr_GMM, Label_GMM, name):
    
    effPriorLogOdds = np.linspace(-4, 4, 100)
    
    dcf_LR = []
    dcf_GMM = []
    dcf_SVM = []
    mindcf_LR = []
    mindcf_SVM = []
    mindcf_GMM = []
    
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        predicted_labels = optimalBinaryBayesDecision(llr_LR, pi, 1, 1)
        conf_matrix = confusionMatrix(Label_LR, predicted_labels)
        dcf_LR.append(DCF(pi, 1, 1, conf_matrix, "normalized"))
        mindcf_LR.append(min_DCF(pi, 1, 1, Label_LR, llr_LR))
    
    print("fine 1")
    
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        predicted_labels = optimalBinaryBayesDecision(llr_SVM, pi, 1, 1)
        conf_matrix = confusionMatrix(Label_SVM, predicted_labels)
        dcf_SVM.append(DCF(pi, 1, 1, conf_matrix, "normalized"))
        mindcf_SVM.append(min_DCF(pi, 1, 1, Label_SVM, llr_SVM))
     
    print("fine 2")   
        
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        predicted_labels = optimalBinaryBayesDecision(llr_GMM, pi, 1, 1)
        conf_matrix = confusionMatrix(Label_GMM, predicted_labels)
        dcf_GMM.append(DCF(pi, 1, 1, conf_matrix, "normalized"))
        mindcf_GMM.append(min_DCF(pi, 1, 1, Label_GMM, llr_GMM))
        
    print("fine 3")   
    plt.plot(effPriorLogOdds, dcf_LR, label="DCF LR", color="red")
    plt.plot(effPriorLogOdds, mindcf_LR, label="min DCF LR", color="red", linestyle="dashed" )
    plt.plot(effPriorLogOdds, dcf_SVM, label="DCF SVM", color="blue")
    plt.plot(effPriorLogOdds, mindcf_SVM, label="min DCF SVM", color="blue", linestyle="dashed" )
    plt.plot(effPriorLogOdds, dcf_GMM, label="DCF GMM", color="green")
    plt.plot(effPriorLogOdds, mindcf_GMM, label="min DCF GMM", color="green", linestyle="dashed" )
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF value')
    plt.title('Bayes error plot')
    plt.legend(loc="upper left")
    plt.savefig("Calibration/Plot/" + name + "_bayes_err_plot.pdf")
    plt.close()