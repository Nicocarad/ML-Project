from Utils.load import load
from Features_Analysis.feature_analysis import *
from Training.Gaussian.train_gaussian import *
import numpy
from Training.Logistic_Regression.train_logistic_regression import *
from Training.SVM.train_SVM import *
from Training.GMM.train_GMM import *
from Calibration.bayes_error_plot import *
from Calibration.calibration import *
from Testing.test import *


if __name__ == '__main__':


# Dataset Load
      D1, L1 = load('Dataset/Train.txt')
      D2, L2 = load('Dataset/Test.txt')
      
# Dataset Info
    
#     num_label_0 = numpy.count_nonzero(L1 == 0)
#     num_label_1 = numpy.count_nonzero(L1 == 1)
#     num_label_0 = numpy.count_nonzero(L2 == 0)
#     num_label_1 = numpy.count_nonzero(L2 == 1)
#     print("(Train) Number of Male", num_label_0)
#     print("(Train) Number of Female", num_label_1)
#     print("(Test) Number of Male", num_label_0)
#     print("(Test) Number of Female", num_label_1)
    
    
# #Feature Analysis

#     plot_centered_hist(D1, L1, "true")
#     plot_scatter(D1,L1)
#     plot_LDA_hist(D1,L1,1)
#     plot_heatmaps_dataset(D1)
#     plot_heatmaps_male(D1,L1)
#     plot_heatmaps_female(D1,L1)
#     PCA_plot(D1)
    
    
# # Train Gaussian Classifiers

#    train_MVG(D1,L1)
#    train_NB(D1,L1)
#    train_TMVG(D1,L1)
#    train_TNB(D1,L1)


# # Train LR

#   LR_RAW(D1,L1,0.5)
#   LR_RAW_Znorm(D1,L1,0.5)
#   LR_PCA(D1,L1,0.5)
#   LR_PCA(D1,L1,0.1)
#   LR_PCA(D1,L1,0.9)
#   LR_diff_priors(D1,L1)
#   Quad_LR_RAW(D1, L1, 0.5)
#   Quad_LR_diff_priors(D1, L1)
#   Quad_LR_diff_priors_Zscore(D1, L1)


# Train SVM

#Lin_SVM(D1,L1)
#DA FARE - (in corso)
# SVM_RAW_znorm_01(D1, L1, 0.5)
# SVM_RAW_znorm_09(D1, L1, 0.5)
# SVM_RAW_znorm_05(D1, L1, 0.5)
#-------------------------------------------
#SVM_diff_priors(D1, L1)
#SVM_diff_priors_znorm(D1, L1)

#DA FARE
# Poly_SVM_RAW_znorm_05(D1, L1, 0.5)
# Poly_SVM_RAW_znorm_01(D1, L1, 0.5)
#Poly_SVM_RAW_znorm_09(D1, L1, 0.5)
#-----------------

#RadKernBased_RAW(D1,L1,0.5)
#RadKernBased_znorm(D1,L1,0.5)



#Poly_SVM_diff_priors(D1, L1)
#Kern_SVM_diff_priors(D1, L1)
#Kern_SVM_diff_priors_znorm(D1, L1)


#TRAIN GMM

# GMM_plot_diff_component(D1, L1)
# GMM_plot_diff_component_PCA(D1,L1,11)
# GMM_plot_diff_component_PCA(D1,L1,10)
# GMM_Tied_plot_diff_component(D1, L1)
# GMM_Tied_plot_diff_component_PCA(D1, L1,11)
# GMM_Tied_plot_diff_component_PCA(D1, L1,10)
# GMM_Diagonal_plot_diff_component(D1, L1)
# GMM_Diagonal_plot_diff_component_PCA(D1, L1)
# GMM_TiedDiagonal_plot_diff_component(D1, L1)

#GMM_diff_priors(D1,L1)
#GMM_diff_priors_zscore(D1,L1)





# CALIBRATION

# llr,Label = GMM_train_best(D1,L1)
# bayes_error_plot(llr,Label,"Best_GMM")

# llr,Label = LR_train_best(D1,L1)
# bayes_error_plot(llr,Label,"Best_LR")

# llr,Label = SVM_train_best(D1,L1)
# bayes_error_plot(llr,Label,"Best_SVM")

# llr_LR,Label_LR = LR_train_best(D1,L1)
# llr_cal_LR,Label_cal_LR = calibration(llr_LR,Label_LR,0.5)
# bayes_error_plot(llr_cal_LR,Label_cal_LR,"Best_LR_calibrated")

# llr_GMM,Label_GMM = GMM_train_best(D1,L1)
# llr_cal_GMM,Label_cal_GMM = calibration(llr_GMM,Label_GMM,0.5)
# bayes_error_plot(llr_cal_GMM,Label_cal_GMM,"Best_GMM_calibrated")

# llr_SVM,Label_SVM = SVM_train_best(D1,L1)
# llr_cal_SVM,Label_cal_SVM = calibration(llr_SVM,Label_SVM,0.5)
# bayes_error_plot(llr_cal_SVM,Label_cal_SVM,"Best_SVM_calibrated")

#EVALUATION

# print("GMM - min_dcf / act_dcf\n")
# llr,Label = GMM_train_best(D1,L1)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)
# print("-----------------------\n")


# print("LR - min_dcf / act_dcf\n")
# llr,Label = LR_train_best(D1,L1)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)

# print("-----------------------\n")
# print("SVM - min_dcf / act_dcf\n")
# llr,Label = SVM_train_best(D1,L1)
# llr_cal,Label_cal = calibration(llr,Label,0.5)
# predicted_labels = optimalBinaryBayesDecision(llr_cal, 0.5, 1, 1)
# conf_matrix = confusionMatrix(Label_cal, predicted_labels)
# min_dcf = min_DCF(0.5,1,1,Label_cal,llr_cal)
# act_dcf = DCF(0.5, 1, 1, conf_matrix, "normalized")
# print("min_dcf: ", min_dcf)
# print("act_dcf: ", act_dcf)



GMM_test_best(D1,D2,L1,L2)


