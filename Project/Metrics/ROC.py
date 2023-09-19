import numpy as np
import matplotlib.pyplot as plt
from Metrics.DCF import *


def calculate_roc_curve(scores, labels):
    thresholds = np.concatenate([scores, [-np.inf, np.inf]])

    thresholds.sort()

    fpr_list = []
    tpr_list = []

    for t in thresholds:
        predicted = (scores > t).astype(int)
        conf_matrix = confusionMatrix(labels, predicted)

        (TN, FN), (FP, TP) = conf_matrix

        fnr = FN / (FN + TP)
        fpr = FP / (FP + TN)

        tpr = 1 - fnr

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return fpr_list, tpr_list


def plot_ROC(scores_LR, LTE_LR, scores_SVM, LTE_SVM, scores_GMM, LTE_GMM):
    fpr_list_LR, tpr_list_LR = calculate_roc_curve(scores_LR, LTE_LR)
    fpr_list_GMM, tpr_list_GMM = calculate_roc_curve(scores_GMM, LTE_GMM)
    fpr_list_SVM, tpr_list_SVM = calculate_roc_curve(scores_SVM, LTE_SVM)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.plot(fpr_list_LR, tpr_list_LR, color="red", label="LR")
    plt.plot(fpr_list_GMM, tpr_list_GMM, color="green", label="GMM")
    plt.plot(fpr_list_SVM, tpr_list_SVM, color="blue", label="SVM")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("Metrics\Plot\ROC.pdf")
    plt.close()
