import matplotlib.pyplot as plt
from Models.SVM.svm import *
from Metrics.DCF import *
from Utils.Kfold import *
from Preprocessing.Znorm import *
from Calibration.calibration import *


def RadKernBased_RAW_eval(DTR, LTR, DTE, LTE, prior):
    C_values = numpy.logspace(-5, 5, num=11)

    min_dcf_results_log1 = []
    min_dcf_results_log2 = []
    min_dcf_results_log3 = []
    min_dcf_results_log1_eval = []
    min_dcf_results_log2_eval = []
    min_dcf_results_log3_eval = []

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.1)

        SPost_1, Label_1 = kfold(svm, 5, DTR, LTR, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log1.append(res_1)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.1)

        svm.train(DTR, LTR, DTE, LTE, 0.5)
        svm.compute_scores()
        scores = svm.scores
        res_1 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log1_eval.append(res_1)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.01)

        SPost_2, Label_2 = kfold(svm, 5, DTR, LTR, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log2.append(res_2)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.01)

        svm.train(DTR, LTR, DTE, LTE, 0.5)
        svm.compute_scores()
        scores = svm.scores

        res_2 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log2_eval.append(res_2)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.001)

        SPost_3, Label_3 = kfold(svm, 5, DTR, LTR, prior)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        min_dcf_results_log3.append(res_3)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.001)

        svm.train(DTR, LTR, DTE, LTE, 0.5)
        svm.compute_scores()
        scores = svm.scores
        res_3 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log3_eval.append(res_3)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("RKB-SVM")

    plt.plot(C_values, min_dcf_results_log1, label="logγ = -1 (VAL)", color="red")
    plt.plot(C_values, min_dcf_results_log2, label="logγ = -2 (VAL)", color="blue")
    plt.plot(C_values, min_dcf_results_log3, label="logγ = -3 (VAL)", color="green")
    plt.plot(
        C_values,
        min_dcf_results_log1_eval,
        label="logγ = -1 (EVAL)",
        color="red",
        linestyle="dashed",
    )
    plt.plot(
        C_values,
        min_dcf_results_log2_eval,
        label="logγ = -2 (EVAL)",
        color="blue",
        linestyle="dashed",
    )
    plt.plot(
        C_values,
        min_dcf_results_log3_eval,
        label="logγ = -3 (EVAL)",
        color="green",
        linestyle="dashed",
    )

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Testing/SVM/Plot/RadKernBased_RAW_eval.pdf")
    plt.close()


def RadKernBased_znorm_eval(DTR, LTR, DTE, LTE, prior):
    C_values = numpy.logspace(-5, 5, num=11)

    min_dcf_results_log1 = []
    min_dcf_results_log2 = []
    min_dcf_results_log3 = []
    min_dcf_results_log1_eval = []
    min_dcf_results_log2_eval = []
    min_dcf_results_log3_eval = []

    DTR = znorm(DTR)
    DTE = znorm(DTE)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.1)

        SPost_1, Label_1 = kfold(svm, 5, DTR, LTR, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log1.append(res_1)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.1)

        svm.train(DTR, LTR, DTE, LTE, 0.5)
        svm.compute_scores()
        scores = svm.scores
        res_1 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log1_eval.append(res_1)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.01)

        SPost_2, Label_2 = kfold(svm, 5, DTR, LTR, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log2.append(res_2)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.01)

        svm.train(DTR, LTR, DTE, LTE, 0.5)
        svm.compute_scores()
        scores = svm.scores

        res_2 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log2_eval.append(res_2)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.001)

        SPost_3, Label_3 = kfold(svm, 5, DTR, LTR, prior)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        min_dcf_results_log3.append(res_3)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.001)

        svm.train(DTR, LTR, DTE, LTE, 0.5)
        svm.compute_scores()
        scores = svm.scores
        res_3 = min_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_log3_eval.append(res_3)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("RKB-SVM")

    plt.plot(C_values, min_dcf_results_log1, label="logγ = -1 (VAL)", color="red")
    plt.plot(C_values, min_dcf_results_log2, label="logγ = -2 (VAL)", color="blue")
    plt.plot(C_values, min_dcf_results_log3, label="logγ = -3 (VAL)", color="green")
    plt.plot(
        C_values,
        min_dcf_results_log1_eval,
        label="logγ = -1 (EVAL)",
        color="red",
        linestyle="dashed",
    )
    plt.plot(
        C_values,
        min_dcf_results_log2_eval,
        label="logγ = -2 (EVAL)",
        color="blue",
        linestyle="dashed",
    )
    plt.plot(
        C_values,
        min_dcf_results_log3_eval,
        label="logγ = -3 (EVAL)",
        color="green",
        linestyle="dashed",
    )

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Testing/SVM/Plot/RadKernBased_znorm_eval.pdf")
    plt.close()


def SVM_test_best(DTR, DTE, LTR, LTE):
    DTR = znorm(DTR)
    DTE = znorm(DTE)
    lbd = 0.1
    C = 5
    pi_T = 0.5

    svm = RadialKernelBasedSvm(1, C, lbd)

    svm.train(DTR, LTR, DTE, LTE, pi_T)
    svm.compute_scores()
    scores = svm.scores

    return scores, LTE


def SVM_test_min_act_dcf_cal(DTR, LTR, DTE, LTE,pi):
    print("SVM - min_dcf / act_dcf " + str(pi) + "\n")
    llr, Label = SVM_test_best(DTR, DTE, LTR, LTE)
    llr_cal, Label_cal = calibration(llr, Label, 0.5)
    predicted_labels = optimalBinaryBayesDecision(llr_cal, pi, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = min_DCF(pi, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(pi, 1, 1, conf_matrix, "normalized")
    print("SVM (test) min_dcf: ", round(min_dcf,3))
    print("SVM (test) act_dcf: ", round(act_dcf,3))
