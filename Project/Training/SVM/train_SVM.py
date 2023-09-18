from Metrics.DCF import min_DCF
from Models.SVM.svm import *
from Utils.Kfold import kfold
import matplotlib.pyplot as plt
from Preprocessing.Znorm import *
from Calibration.calibration import *
from Metrics.DCF import *
import matplotlib.pyplot as plt


def SVM_RAW_znorm(D, L, prior, pi):
    C_values = numpy.logspace(-5, 5, num=11)

    min_dcf_results = []
    min_dcf_results_znorm = []

    for i, c in enumerate(C_values):
        svm = Linear_SVM(1, c)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(pi, 1, 1, Label_1, SPost_1)
        print(res_1)
        min_dcf_results.append(res_1)
        print(i)

    D = znorm(D)
    for i, c in enumerate(C_values):
        svm = Linear_SVM(1, c)
        

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(pi, 1, 1, Label_2, SPost_2)
        print(res_2)
        min_dcf_results_znorm.append(res_2)
        print(i)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")

    plt.plot(C_values, min_dcf_results, label="minDCF(\u03C0 = " + str(pi) + ") RAW")
    plt.plot(
        C_values, min_dcf_results_znorm, label="minDCF(\u03C0 = " + str(pi) + ") Z-norm"
    )

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/Lin_SVM_RAW_Znorm_" + str(pi) + ".pdf")
    plt.close()


def SVM_diff_priors(D, L):
    C = 10
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
        svm = Linear_SVM(1, C)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def SVM_diff_priors_znorm(D, L):
    C = 10
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
        svm = Linear_SVM(1, C)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF_znorm (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")




def Poly_SVM_RAW_znorm(D, L, prior, pi):
    C_values = numpy.logspace(-5, 5, num=2)

    min_dcf_results = []
    min_dcf_results_znorm = []

    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1, c, 2, 1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(pi, 1, 1, Label_1, SPost_1)
        print(res_1)
        min_dcf_results.append(res_1)
        print(i)

    D = znorm(D)
    for i, c in enumerate(C_values):
        svm = PolynomialSvm(1, c, 2, 1)
        

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(pi, 1, 1, Label_2, SPost_2)
        print(res_2)
        min_dcf_results_znorm.append(res_2)
        print(i)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")

    plt.plot(C_values, min_dcf_results, label="minDCF(\u03C0 = " + str(pi) + ") RAW")
    plt.plot(
        C_values, min_dcf_results_znorm, label="minDCF(\u03C0 = " + str(pi) + ") Z-norm"
    )

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/TEST_Poly_SVM_RAW_Znorm_" + str(pi) + ".pdf")
    plt.close()


def Poly_SVM_diff_priors_znorm(D, L):
    C = 10
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
        svm = PolynomialSvm(1, C, 2, 1)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def Poly_SVM_diff_priors(D, L):
    C = 0.001
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
        svm = PolynomialSvm(1, C, 2, 1)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")







def RadKernBased_RAW(D, L, prior):
    C_values = numpy.logspace(-5, 5, num=11)

    min_dcf_results_log1 = []
    min_dcf_results_log2 = []
    min_dcf_results_log3 = []

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log1.append(res_1)
        print(i)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.01)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log2.append(res_2)
        print(i)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.001)

        SPost_3, Label_3 = kfold(svm, 5, D, L, prior)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        min_dcf_results_log3.append(res_3)
        print(i)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")

    plt.plot(C_values, min_dcf_results_log1, label="logγ = -1")
    plt.plot(C_values, min_dcf_results_log2, label="logγ = -2")
    plt.plot(C_values, min_dcf_results_log3, label="logγ = -3")

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/RadKernBased_RAW.pdf")
    plt.close()


def RadKernBased_znorm(D, L, prior):
    C_values = numpy.logspace(-5, 5, num=11)

    D = znorm(D)
    min_dcf_results_log1 = []
    min_dcf_results_log2 = []
    min_dcf_results_log3 = []

    for i, c in enumerate(C_values):
        print(c)
        svm = RadialKernelBasedSvm(1, c, 0.1)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log1.append(res_1)
        print(i)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.01)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log2.append(res_2)
        print(i)

    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.001)

        SPost_3, Label_3 = kfold(svm, 5, D, L, prior)
        res_3 = min_DCF(0.5, 1, 1, Label_3, SPost_3)
        min_dcf_results_log3.append(res_3)
        print(i)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")

    plt.plot(C_values, min_dcf_results_log1, label="logγ = -1")
    plt.plot(C_values, min_dcf_results_log2, label="logγ = -2")
    plt.plot(C_values, min_dcf_results_log3, label="logγ = -3")

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/RadKernBased_znorm.pdf")
    plt.close()


def RadKernBased_diff_priors(D, L):
    C = 10
    lbd = 0.001
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
        svm = RadialKernelBasedSvm(1, C, lbd)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def RadKernBased_diff_priors_znorm(D, L):
    C = 5
    lbd = 0.1
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
        svm = RadialKernelBasedSvm(1, C, lbd)
        SPost, Label = kfold(svm, 5, D, L, pi_T)
        res = min_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def RadKernBased_RAW_znorm(D, L, prior):
    C_values = numpy.logspace(-1, 1, num=5)

    min_dcf_results_log1_znorm = []

    min_dcf_results_log3_raw = []

    for i, c in enumerate(C_values):
        print(c)
        svm = RadialKernelBasedSvm(1, c, 0.001)

        SPost_1, Label_1 = kfold(svm, 5, D, L, prior)
        res_1 = min_DCF(0.5, 1, 1, Label_1, SPost_1)
        min_dcf_results_log3_raw.append(res_1)
        print(i)

    D = znorm(D)
    for i, c in enumerate(C_values):
        svm = RadialKernelBasedSvm(1, c, 0.1)

        SPost_2, Label_2 = kfold(svm, 5, D, L, prior)
        res_2 = min_DCF(0.5, 1, 1, Label_2, SPost_2)
        min_dcf_results_log1_znorm.append(res_2)
        print(i)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")

    plt.plot(C_values, min_dcf_results_log1_znorm, label="logγ = -1 Znorm")
    plt.plot(C_values, min_dcf_results_log3_raw, label="logγ = -3 RAW")

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig("Training/SVM/Plot/RadKernBased_raw_znorm.pdf")
    plt.close()


def SVM_train_best(D, L):
    D = znorm(D)
    lbd = 0.1
    C = 5
    pi_T = 0.5

    svm = RadialKernelBasedSvm(1, C, lbd)
    SPost, Label = kfold(svm, 5, D, L, pi_T)
    return SPost, Label




def SVM_min_act_dcf_cal(DTR,LTR):
    print("SVM - min_dcf / act_dcf\n")
    llr,Label = SVM_train_best(DTR,LTR)
    llr_cal,Label_cal = calibration(llr,Label,0.5)
    predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
    act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
    print("min_dcf: ", min_dcf)
    print("act_dcf: ", act_dcf)