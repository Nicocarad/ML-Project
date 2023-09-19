import matplotlib.pyplot as plt
import numpy as np

from Models.LR.logistic_regression import *
from Metrics.DCF import *
from Utils.Kfold import *
from Preprocessing.Znorm import *
from Calibration.calibration import *


def plot_RAW_results(
    min_dcf_05,
    min_dcf_01,
    min_dcf_09,
    min_dcf_05_eval,
    min_dcf_01_eval,
    min_dcf_09_eval,
    name,
    title,
):
    lambda_values = np.logspace(-5, 2, num=41)

    plt.figure()
    plt.xlabel("\u03BB")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title(title)

    plt.plot(lambda_values, min_dcf_05, label="minDCF(\u03C0 = 0.5 VAL)", color="red")
    plt.plot(lambda_values, min_dcf_01, label="minDCF(\u03C0 = 0.1 VAL)", color="blue")
    plt.plot(lambda_values, min_dcf_09, label="minDCF(\u03C0 = 0.9 VAL)", color="green")
    plt.plot(
        lambda_values,
        min_dcf_05_eval,
        label="minDCF(\u03C0 = 0.5 EVAL)",
        color="red",
        linestyle="dashed",
    )
    plt.plot(
        lambda_values,
        min_dcf_01_eval,
        label="minDCF(\u03C0 = 0.1 EVAL)",
        color="blue",
        linestyle="dashed",
    )
    plt.plot(
        lambda_values,
        min_dcf_09_eval,
        label="minDCF(\u03C0 = 0.9 EVAL)",
        color="green",
        linestyle="dashed",
    )

    plt.xlim(lambda_values[0], lambda_values[-1])
    plt.legend()

    plt.savefig("Testing/Logistic_Regression/Plot/" + name + ".pdf")
    plt.close()


def LR_RAW_val_eval(DTR, LTR, DTE, LTE, prior):
    name = "LR_RAW_val_eval"
    l_values = np.logspace(-5, 2, num=41)
    value = [0.5, 0.1, 0.9]

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []
    min_dcf_results_05_eval = []
    min_dcf_results_01_eval = []
    min_dcf_results_09_eval = []

    regression = Logistic_Regression(0)
    reg_eval = Logistic_Regression(0)

    for i, l in enumerate(l_values):
        regression.l = l
        reg_eval.l = l

        SPost, Label = kfold(regression, 5, DTR, LTR, prior)
        min_dcf_results_05.append(min_DCF(value[0], 1, 1, Label, SPost))
        min_dcf_results_01.append(min_DCF(value[1], 1, 1, Label, SPost))
        min_dcf_results_09.append(min_DCF(value[2], 1, 1, Label, SPost))

        reg_eval.train(DTR, LTR, DTE, LTE, 0.5)
        reg_eval.compute_scores()
        scores = reg_eval.scores
        min_dcf_results_05_eval.append(min_DCF(value[0], 1, 1, LTE, scores))
        min_dcf_results_01_eval.append(min_DCF(value[1], 1, 1, LTE, scores))
        min_dcf_results_09_eval.append(min_DCF(value[2], 1, 1, LTE, scores))

        print(i)

    plot_RAW_results(
        min_dcf_results_05,
        min_dcf_results_01,
        min_dcf_results_09,
        min_dcf_results_05_eval,
        min_dcf_results_01_eval,
        min_dcf_results_09_eval,
        name,
        "RAW",
    )


def LR_znorm_val_eval(DTR, LTR, DTE, LTE, prior):
    name = "LR_znorm_val_eval"
    l_values = np.logspace(-5, 2, num=41)
    value = [0.5, 0.1, 0.9]

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []
    min_dcf_results_05_eval = []
    min_dcf_results_01_eval = []
    min_dcf_results_09_eval = []

    regression = Logistic_Regression(0)
    reg_eval = Logistic_Regression(0)

    DTR = znorm(DTR)
    DTE = znorm(DTE)

    for i, l in enumerate(l_values):
        regression.l = l
        reg_eval.l = l

        SPost, Label = kfold(regression, 5, DTR, LTR, prior)
        min_dcf_results_05.append(min_DCF(value[0], 1, 1, Label, SPost))
        min_dcf_results_01.append(min_DCF(value[1], 1, 1, Label, SPost))
        min_dcf_results_09.append(min_DCF(value[2], 1, 1, Label, SPost))

        reg_eval.train(DTR, LTR, DTE, LTE, 0.5)
        reg_eval.compute_scores()
        scores = reg_eval.scores
        min_dcf_results_05_eval.append(min_DCF(value[0], 1, 1, LTE, scores))
        min_dcf_results_01_eval.append(min_DCF(value[1], 1, 1, LTE, scores))
        min_dcf_results_09_eval.append(min_DCF(value[2], 1, 1, LTE, scores))

        print(i)

    plot_RAW_results(
        min_dcf_results_05,
        min_dcf_results_01,
        min_dcf_results_09,
        min_dcf_results_05_eval,
        min_dcf_results_01_eval,
        min_dcf_results_09_eval,
        name,
        "Z-NORM",
    )


def LR_test_best(DTR, DTE, LTR, LTE):
    l = 0.01
    pi_T = 0.1
    regression = Logistic_Regression(l)
    regression.train(DTR, LTR, DTE, LTE, pi_T)
    regression.compute_scores()
    scores = regression.scores

    return scores, LTE


def LR_test_min_act_dcf_cal(DTR, LTR, DTE, LTE,pi):
    print("LR - min_dcf / act_dcf " + str(pi) + "\n")
    llr, Label = LR_test_best(DTR, DTE, LTR, LTE)
    llr_cal, Label_cal = calibration(llr, Label, 0.5)
    predicted_labels = optimalBinaryBayesDecision(llr_cal, pi, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = min_DCF(pi, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(pi, 1, 1, conf_matrix, "normalized")
    print("LR (test) min_dcf: ", round(min_dcf,3))
    print("LR (test) act_dcf: ", round(act_dcf,3))
