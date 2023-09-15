from Utils.load import load
from Features_Analysis.feature_analysis import *
from Training.Gaussian.train_gaussian import *
from Training.Logistic_Regression.train_logistic_regression import *
from Training.SVM.train_SVM import *
from Training.GMM.train_GMM import *
from Metrics.bayes_error_plot import *
from Calibration.calibration import *
from Testing.test import *
from Testing.Logistic_Regression.LR_test import *
from Testing.GMM.GMM_test import *
from Testing.SVM.SVM_test import *

from Metrics.ROC import *


if __name__ == '__main__':


# Dataset Load
      DTR, LTR = load('Dataset/Train.txt')
      DTE, LTE = load('Dataset/Test.txt')
      
# Dataset Info
    
# num_label_0 = numpy.count_nonzero(LTR == 0)
# num_label_1 = numpy.count_nonzero(LTR == 1)
# num_label_0 = numpy.count_nonzero(LTE == 0)
# num_label_1 = numpy.count_nonzero(LTE == 1)
# print("(Train) Number of Male", num_label_0)
# print("(Train) Number of Female", num_label_1)
# print("(Test) Number of Male", num_label_0)
# print("(Test) Number of Female", num_label_1)
    
    
# #Feature Analysis

#     plot_centered_hist(DTR, LTR, "true")
#     plot_scatter(DTR,LTR)
#     plot_LDA_hist(DTR,LTR,1)
#     plot_heatmaps_dataset(DTR)
#     plot_heatmaps_male(DTR,LTR)
#     plot_heatmaps_female(DTR,LTR)
#     PCA_plot(DTR)
    
    
# # Train Gaussian Classifiers

# train_MVG(DTR,LTR)
# train_NB(DTR,LTR)
# train_TMVG(DTR,LTR)
# train_TNB(DTR,LTR)


# # Train LR

#   LR_RAW(DTR,LTR,0.5)
#   LR_RAW_Znorm(DTR,LTR,0.5)
#   LR_PCA(DTR,LTR,0.5)
#   LR_PCA(DTR,LTR,0.1)
#   LR_PCA(DTR,LTR,0.9)
#   LR_diff_priors(DTR,LTR)
#   LR_diff_priors_zscore(DTR, LTR)
#   Quad_LR_RAW(DTR, LTR, 0.5)
#   Quad_LR_diff_priors(DTR, LTR)
#   Quad_LR_diff_priors_Zscore(DTR, LTR)


# Train SVM

#Lin_SVM(DTR,LTR)
#DA FARE - (in corso)
# SVM_RAW_znorm_01(DTR, LTR, 0.5)
# SVM_RAW_znorm_09(DTR, LTR, 0.5)
# SVM_RAW_znorm_05(DTR, LTR, 0.5)
#SVM_diff_priors(DTR, LTR)
#SVM_diff_priors_znorm(DTR, LTR)


# Poly_SVM_RAW_znorm_05(DTR, LTR, 0.5)
# Poly_SVM_RAW_znorm_01(DTR, LTR, 0.5)
# Poly_SVM_RAW_znorm_09(DTR, LTR, 0.5)


#RadKernBased_RAW(DTR,LTR,0.5)
#RadKernBased_znorm(DTR,LTR,0.5)



# Poly_SVM_diff_priors(DTR, LTR)
# Kern_SVM_diff_priors(DTR, LTR)
# Kern_SVM_diff_priors_znorm(DTR, LTR)


#TRAIN GMM

# GMM_plot_diff_component(DTR, LTR)
# GMM_plot_diff_component_PCA(DTR,LTR,11)
# GMM_plot_diff_component_PCA(DTR,LTR,10)
# GMM_Tied_plot_diff_component(DTR, LTR)
# GMM_Tied_plot_diff_component_PCA(DTR, LTR,11)
# GMM_Tied_plot_diff_component_PCA(DTR, LTR,10)
# GMM_Diagonal_plot_diff_component(DTR, LTR)
# GMM_Diagonal_plot_diff_component_PCA(DTR, LTR)
# GMM_TiedDiagonal_plot_diff_component(DTR, LTR)

# GMM_diff_priors(DTR,LTR)
# GMM_diff_priors_zscore(DTR,LTR)





# CALIBRATION (VAL)

#llr,Label = GMM_train_best(DTR,LTR)
# bayes_error_plot(llr,Label,"Best_GMM")

# llr,Label = LR_train_best(DTR,LTR)
# bayes_error_plot(llr,Label,"Best_LR")

# llr,Label = SVM_train_best(DTR,LTR)
# bayes_error_plot(llr,Label,"Best_SVM")

# llr_LR,Label_LR = LR_train_best(DTR,LTR)
# llr_cal_LR,Label_cal_LR = calibration(llr_LR,Label_LR,0.5)
# bayes_error_plot(llr_cal_LR,Label_cal_LR,"Best_LR_calibrated")

# llr_GMM,Label_GMM = GMM_train_best(DTR,LTR)
# llr_cal_GMM,Label_cal_GMM = calibration(llr_GMM,Label_GMM,0.5)
# bayes_error_plot(llr_cal_GMM,Label_cal_GMM,"Best_GMM_calibrated")

# llr_SVM,Label_SVM = SVM_train_best(DTR,LTR)
# llr_cal_SVM,Label_cal_SVM = calibration(llr_SVM,Label_SVM,0.5)
# bayes_error_plot(llr_cal_SVM,Label_cal_SVM,"Best_SVM_calibrated")






#EVALUATION

# print("GMM - min_dcf / act_dcf\n")
# llr,Label = GMM_train_best(DTR,LTR)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)
# print("-----------------------\n")


# print("LR - min_dcf / act_dcf\n")
# llr,Label = LR_train_best(DTR,LTR)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)

# print("-----------------------\n")
# print("SVM - min_dcf / act_dcf\n")
# llr,Label = SVM_train_best(DTR,LTR)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)



# print("GMM - min_dcf / act_dcf\n")
# llr,Label = GMM_test_best(DTR,DTE,LTR,LTE)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)
# print("-----------------------\n")


# print("LR - min_dcf / act_dcf\n")
# llr,Label = LR_test_best(DTR,DTE,LTR,LTE)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)


# print("-----------------------\n")
# print("SVM - min_dcf / act_dcf\n")
# llr,Label = SVM_test_best(DTR,DTE,LTR,LTE)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)






# CALIBRATION (EVAL)

# llr,Label = GMM_test_best(DTR,DTE,LTR,LTE)
# bayes_error_plot(llr,Label,"Best_eval_GMM")

# llr,Label = LR_test_best(DTR,DTE,LTR,LTE)
# bayes_error_plot(llr,Label,"Best_eval_LR")

# llr,Label = SVM_test_best(DTR,DTE,LTR,LTE)
# bayes_error_plot(llr,Label,"Best_eval_SVM")

# llr_LR,Label_LR = LR_test_best(DTR,DTE,LTR,LTE)
# llr_cal_LR,Label_cal_LR = calibration(llr_LR,Label_LR,0.5)


# llr_GMM,Label_GMM = GMM_test_best(DTR,DTE,LTR,LTE)
# llr_cal_GMM,Label_cal_GMM = calibration(llr_GMM,Label_GMM,0.5)


# llr_SVM,Label_SVM = SVM_test_best(DTR,DTE,LTR,LTE)
# llr_cal_SVM,Label_cal_SVM = calibration(llr_SVM,Label_SVM,0.5)


#plot_ROC(llr_cal_LR,Label_cal_LR,llr_cal_SVM,Label_cal_SVM,llr_cal_GMM,Label_cal_GMM)
#plot_DET(llr_cal_LR,Label_cal_LR,llr_cal_SVM,Label_cal_SVM,llr_cal_GMM,Label_cal_GMM,)
#plot_det(llr_cal_LR,Label_cal_LR)
#
#bayes_error_plot_multi(llr_cal_LR, Label_cal_LR, llr_cal_SVM, Label_cal_SVM, llr_cal_GMM, Label_cal_GMM, "Multi_model")



# Testing LR

# LR_RAW_val_eval(DTR, LTR, DTE, LTE, 0.5)
# LR_znorm_val_eval(DTR, LTR, DTE, LTE, 0.5)


# Testing SVM
#RadKernBased_RAW_eval(DTR,LTR,DTE,LTE,0.5)
#RadKernBased_znorm_eval(DTR,LTR,DTE,LTE,0.5)

# Testing GMM

#GMM_plot_diff_component_eval(DTR,LTR,DTE,LTE)
#GMM_plot_diff_component_eval_PCA(DTR, LTR, DTE, LTE)
# GMM_Tied_plot_diff_component_eval_PCA(DTR, LTR, DTE, LTE)
# GMM_Diagonal_plot_diff_component_eval_znorm(DTR, LTR, DTE, LTE)
#GMM_test_compare(DTR, LTR, DTE, LTE)


# plot_ROC(llr_cal_LR, Label_cal_LR, llr_cal_SVM, Label_cal_SVM, llr_cal_GMM, Label_cal_GMM)