import numpy as np
import matplotlib.pyplot as plt
from Utils.Kfold import kfold
from Metrics.DCF import min_DCF
from Models.LR.logistic_regression import *
from Preprocessing.PCA import PCA
from Preprocessing.Znorm import *
from Calibration.calibration import *
from Metrics.DCF import *


def plot_results(min_dcf_05, min_dcf_01, min_dcf_09, name, title):
    lambda_values = np.logspace(-5, 2, num=41)
    plt.figure()
    plt.xlabel("\u03BB")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title(title)

    plt.plot(lambda_values, min_dcf_05, label="minDCF(\u03C0 = 0.5)")
    plt.plot(lambda_values, min_dcf_01, label="minDCF(\u03C0 = 0.1)")
    plt.plot(lambda_values, min_dcf_09, label="minDCF(\u03C0 = 0.9)")
    plt.xlim(lambda_values[0], lambda_values[-1])
    plt.legend()

    plt.savefig("Training/LR/Plot/" + name + ".pdf")
    plt.close()


def LR_RAW(D, L, prior):
    l_values = np.logspace(-5, 2, num=41)
    value = [0.5, 0.1, 0.9]
    name = "LR_RAW"

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    regression = Logistic_Regression(0)

    for i, l in enumerate(l_values):
        regression.l = l
        SPost, Label = kfold(regression, 5, D, L, prior)

        res = min_DCF(value[0], 1, 1, Label, SPost)
        min_dcf_results_05.append(res)

        res = min_DCF(value[1], 1, 1, Label, SPost)
        min_dcf_results_01.append(res)

        res = min_DCF(value[2], 1, 1, Label, SPost)
        min_dcf_results_09.append(res)

        print(i)

    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "RAW"
    )


def LR_RAW_priors(D, L, prior):
    l_values = np.logspace(-5, 2, num=41)
    value = [0.5, 0.1, 0.9]
    name = "LR_RAW_priors"

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    regression = Logistic_Regression(0)

    for i, l in enumerate(l_values):
        regression.l = l

        SPost_1, Label_1 = kfold(regression, 5, D, L, value[0])
        res_1 = min_DCF(prior, 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)

        SPost_2, Label_2 = kfold(regression, 5, D, L, value[1])
        res_2 = min_DCF(prior, 1, 1, Label_2, SPost_2)
        min_dcf_results_01.append(res_2)

        SPost_3, Label_3 = kfold(regression, 5, D, L, value[2])
        res_3 = min_DCF(prior, 1, 1, Label_3, SPost_3)
        min_dcf_results_09.append(res_3)

        print(i)

    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "RAW"
    )


def LR_Znorm(D, L, prior):
    D = znorm(D)
    name = "LR_znorm"
    l_values = np.logspace(-5, 2, num=41)

    value = [0.5, 0.1, 0.9]

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    regression = Logistic_Regression(0)

    for i, l in enumerate(l_values):
        regression.l = l

        SPost, Label = kfold(regression, 5, D, L, prior)
        res = min_DCF(value[0], 1, 1, Label, SPost)
        min_dcf_results_05.append(res)

        res = min_DCF(value[1], 1, 1, Label, SPost)
        min_dcf_results_01.append(res)

        res = min_DCF(value[2], 1, 1, Label, SPost)
        min_dcf_results_09.append(res)

        print(i)
    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "Z-NORM"
    )


def LR_Znorm_priors(D, L, value):
    D = znorm(D)
    name = "LR_znorm_priors"
    l_values = np.logspace(-5, 2, num=41)

    priors = [0.5, 0.1, 0.9]

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    regression = Logistic_Regression(0)
    for i, l in enumerate(l_values):
        regression.l = l

        SPost_1, Label_1 = kfold(regression, 5, D, L, priors[0])
        res_1 = min_DCF(value, 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)

        SPost_2, Label_2 = kfold(regression, 5, D, L, priors[1])
        res_2 = min_DCF(value, 1, 1, Label_2, SPost_2)
        min_dcf_results_01.append(res_2)

        SPost_3, Label_3 = kfold(regression, 5, D, L, priors[2])
        res_3 = min_DCF(value, 1, 1, Label_3, SPost_3)
        min_dcf_results_09.append(res_3)

        print(i)
    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "Z-NORM"
    )


def LR_PCA(D, L, pi_T):
    name = "LR_PCA"
    l_values = np.logspace(-5, 2, num=41)

    value = 0.5

    min_dcf_results_raw = []
    min_dcf_results_11 = []
    min_dcf_results_10 = []
    min_dcf_results_09 = []

    for l in l_values:
        regression = Logistic_Regression(l)
        SPost_1, Label_1 = kfold(regression, 5, D, L, pi_T)
        res_1 = min_DCF(value, 1, 1, Label_1, SPost_1)
        min_dcf_results_raw.append(res_1)
    print("END1")
    pca_dimensions = [11, 10, 9]
    min_dcf_results = {}

    for dim in pca_dimensions:
        D_pca, _ = PCA(D, dim)
        min_dcf_results[dim] = []

        for l in l_values:
            regression = Logistic_Regression(l)
            SPost, Label = kfold(regression, 5, D_pca, L, pi_T)
            res = min_DCF(value, 1, 1, Label, SPost)
            min_dcf_results[dim].append(res)

    min_dcf_results_11 = min_dcf_results[11]
    min_dcf_results_10 = min_dcf_results[10]
    min_dcf_results_09 = min_dcf_results[9]

    plt.figure()
    plt.xlabel("\u03BB")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("RAW + PCA")

    plt.plot(l_values, min_dcf_results_raw, label="minDCF(\u03C0 = 0.5) RAW")
    plt.plot(l_values, min_dcf_results_11, label="minDCF(\u03C0 = 0.5) PCA 11")
    plt.plot(l_values, min_dcf_results_10, label="minDCF(\u03C0 = 0.5) PCA 10")
    plt.plot(l_values, min_dcf_results_09, label="minDCF(\u03C0 = 0.5) PCA 9")

    plt.xlim(l_values[0], l_values[-1])
    plt.legend()

    plt.savefig("Training/LR/Plot/" + name + "_" + str(pi_T) + ".pdf")
    plt.close()


def LR_diff_priors(D, L):
    l = 0.01
    priors = [
        (0.5, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.9, 0.9),
    ]

    for pi_T, pi in priors:
        regression = Logistic_Regression(l)
        SPost, Label = kfold(regression, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def LR_diff_priors_zscore(D, L):
    l = 0.0001
    priors = [
        (0.5, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.9, 0.9),
    ]
    D = znorm(D)
    for pi_T, pi in priors:
        regression = Logistic_Regression(l)
        SPost, Label = kfold(regression, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF znorm (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def Quad_LR_RAW(D, L, prior):
    l_values = np.logspace(-5, 5, num=31)

    value = [0.5]

    min_dcf_results_05 = []
    min_dcf_results_05_znorm = []

    for i, l in enumerate(l_values):
        regression = Quad_Logistic_Regression(l)

        SPost_1, Label_1 = kfold(regression, 5, D, L, prior)
        res_1 = min_DCF(value[0], 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)
        print(i)

    D = znorm(D)
    for i, l in enumerate(l_values):
        regression = Quad_Logistic_Regression(l)

        SPost_2, Label_2 = kfold(regression, 5, D, L, prior)
        res_2 = min_DCF(value[0], 1, 1, Label_2, SPost_2)
        min_dcf_results_05_znorm.append(res_2)
        print(i)

    plt.figure()
    plt.xlabel("\u03BB")
    plt.xscale("log")
    plt.ylabel("minDCF")

    plt.plot(l_values, min_dcf_results_05, label="minDCF(\u03C0 = 0.5) RAW")
    plt.plot(l_values, min_dcf_results_05_znorm, label="minDCF(\u03C0 = 0.5) Z-norm")

    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig("Training/LR/Plot/Quad_LR.pdf")
    plt.close()


def Quad_LR_diff_priors(D, L):
    l = 100
    priors = [
        (0.5, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.9, 0.9),
    ]

    for pi_T, pi in priors:
        regression = Quad_Logistic_Regression(l)
        SPost, Label = kfold(regression, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def Quad_LR_diff_priors_zscore(D, L):
    l = 0
    priors = [
        (0.5, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.9, 0.9),
    ]
    D = znorm(D)
    for pi_T, pi in priors:
        regression = Quad_Logistic_Regression(l)
        SPost, Label = kfold(regression, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF_znorm (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def LR_train_best(D, L):
    l = 0.01
    regression = Logistic_Regression(l)
    SPost, Label = kfold(regression, 5, D, L, 0.1)
    return SPost, Label


def LR_min_act_dcf_cal(DTR, LTR, pi):
    print("LR - min_dcf / act_dcf " + str(pi) + "\n")
    llr, Label = LR_train_best(DTR, LTR)
    llr_cal, Label_cal = calibration(llr, Label, 0.5)
    predicted_labels = optimalBinaryBayesDecision(llr_cal, pi, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = min_DCF(pi, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(pi, 1, 1, conf_matrix, "normalized")
    print("LR (train) min_dcf: ", round(min_dcf, 3))
    print("LR (train) act_dcf: ", round(act_dcf, 3))
