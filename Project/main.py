from Utils.load import *
from Features_Analysis.feature_analysis import *
from Training.Gaussian.train_gaussian import *
from Training.LR.train_logistic_regression import *
from Training.SVM.train_SVM import *
from Training.GMM.train_GMM import *
from Metrics.bayes_error_plot import *
from Calibration.calibration import *
from Testing.Logistic_Regression.LR_test import *
from Testing.GMM.GMM_test import *
from Testing.SVM.SVM_test import *
from Metrics.ROC import *


if __name__ == '__main__':


# DATASET LOAD

      DTR, LTR = load('Dataset/Train.txt')
      DTE, LTE = load('Dataset/Test.txt')
      
# # DATASET INFO
    
      # num_label_0 = numpy.count_nonzero(LTR == 0)
      # num_label_1 = numpy.count_nonzero(LTR == 1)
      # num_label_0 = numpy.count_nonzero(LTE == 0)
      # num_label_1 = numpy.count_nonzero(LTE == 1)
      # print("(Train) Number of Male", num_label_0)
      # print("(Train) Number of Female", num_label_1)
      # print("(Test) Number of Male", num_label_0)
      # print("(Test) Number of Female", num_label_1)
  
  
#----------------------------------------------------------------------------------------#  
    
    
# # FEATURES ANALYSIS

      # plot_centered_hist(DTR, LTR, "true")
      # plot_scatter(DTR,LTR)
      # plot_LDA_hist(DTR,LTR,1)
      # plot_heatmaps_dataset(DTR)
      # plot_heatmaps_male(DTR,LTR)
      # plot_heatmaps_female(DTR,LTR)
      # PCA_plot(DTR)
   
    
#----------------------------------------------------------------------------------------#    


# # TRAIN GAUSSIAN CLASSIFIER

      # train_MVG(DTR,LTR)
      # train_NB(DTR,LTR)
      # train_TMVG(DTR,LTR)
      # train_TNB(DTR,LTR)


#----------------------------------------------------------------------------------------# 


# # TRAIN LOGISTIC REGRESSION

      # LR_RAW(DTR,LTR,0.5)
      # LR_Znorm(DTR,LTR,0.5)
      # LR_RAW_priors(DTR, LTR, 0.5)
      # LR_Znorm_priors(DTR, LTR, 0.5)
      # LR_PCA(DTR,LTR,0.5)
      # LR_PCA(DTR,LTR,0.1)
      # LR_PCA(DTR,LTR,0.9)
      # LR_diff_priors(DTR,LTR)
      # LR_diff_priors_zscore(DTR, LTR)

      # Quad_LR_RAW(DTR, LTR, 0.5)
      # Quad_LR_diff_priors(DTR, LTR)
      # Quad_LR_diff_priors_zscore(DTR, LTR)


#----------------------------------------------------------------------------------------# 


# # TRAIN SUPPORT VECTOR MACHINE


      # SVM_RAW_znorm(DTR, LTR, 0.5, 0.5)
      # SVM_RAW_znorm(DTR, LTR, 0.5, 0.1)
      # SVM_RAW_znorm(DTR, LTR, 0.5, 0.9)
      # SVM_diff_priors(DTR, LTR)
      # SVM_diff_priors_znorm(DTR, LTR)

      # Poly_SVM_RAW_znorm(DTR, LTR, 0.5, 0.5)
      # Poly_SVM_RAW_znorm(DTR, LTR, 0.5, 0.1)
      # Poly_SVM_RAW_znorm(DTR, LTR, 0.5, 0.9)
      # Poly_SVM_diff_priors(DTR, LTR)
      # Poly_SVM_diff_priors_znorm(DTR, LTR)

      # RadKernBased_RAW(DTR,LTR,0.5)
      # RadKernBased_znorm(DTR,LTR,0.5)
      # RadKernBased_RAW_znorm(DTR, LTR, 0.5)
      # RadKernBased_diff_priors(DTR, LTR)
      # RadKernBased_diff_priors_znorm(DTR, LTR)


#----------------------------------------------------------------------------------------# 


# # TRAIN GAUSSIAN MIXTURE MODEL

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
      # GMM_diff_priors_znorm(DTR,LTR)


#----------------------------------------------------------------------------------------# 


# # CALIBRATION (VAL)

      # llr,Label = GMM_train_best(DTR,LTR)
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


#----------------------------------------------------------------------------------------# 


# # EVALUATION

      # LR_min_act_dcf_cal(DTR,LTR,0.5)
      # SVM_min_act_dcf_cal(DTR,LTR,0.5)
      # GMM_min_act_dcf_cal(DTR,LTR,0.5)

      # LR_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.5)
      # SVM_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.5)
      # GMM_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.5)
      
      
      # LR_min_act_dcf_cal(DTR,LTR,0.1)
      # SVM_min_act_dcf_cal(DTR,LTR,0.1)
      # GMM_min_act_dcf_cal(DTR,LTR,0.1)

      # LR_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.1)
      # SVM_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.1)
      # GMM_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.1)
      
      
      # LR_min_act_dcf_cal(DTR,LTR,0.9)
      # SVM_min_act_dcf_cal(DTR,LTR,0.9)
      # GMM_min_act_dcf_cal(DTR,LTR,0.9)

      # LR_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.9)
      # SVM_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.9)
      # GMM_test_min_act_dcf_cal(DTR,LTR,DTE,LTE,0.9)


#----------------------------------------------------------------------------------------# 


# # CALIBRATION (EVAL)

      # llr_LR,Label_LR = LR_test_best(DTR,DTE,LTR,LTE)
      # llr_cal_LR,Label_cal_LR = calibration(llr_LR,Label_LR,0.5)


      # llr_GMM,Label_GMM = GMM_test_best(DTR,DTE,LTR,LTE)
      # llr_cal_GMM,Label_cal_GMM = calibration(llr_GMM,Label_GMM,0.5)


      # llr_SVM,Label_SVM = SVM_test_best(DTR,DTE,LTR,LTE)
      # llr_cal_SVM,Label_cal_SVM = calibration(llr_SVM,Label_SVM,0.5)


      # plot_ROC(llr_cal_LR,Label_cal_LR,llr_cal_SVM,Label_cal_SVM,llr_cal_GMM,Label_cal_GMM)
      # bayes_error_plot_multi(llr_cal_LR, Label_cal_LR, llr_cal_SVM, Label_cal_SVM, llr_cal_GMM, Label_cal_GMM, "Multi_model")


#----------------------------------------------------------------------------------------# 


# # TEST LOGISTIC REGRESSION

      # LR_RAW_val_eval(DTR, LTR, DTE, LTE, 0.1)
      # LR_znorm_val_eval(DTR, LTR, DTE, LTE, 0.1)
      # LR_RAW_priors(DTR, LTR, DTE, LTE, 0.5)
      # LR_znorm_priors(DTR, LTR, DTE, LTE, 0.5)
#----------------------------------------------------------------------------------------# 


# # TESTING SUPPORT VECTOR MACHINE

      # RadKernBased_RAW_eval(DTR,LTR,DTE,LTE,0.5)
      # RadKernBased_znorm_eval(DTR,LTR,DTE,LTE,0.5)
      

#----------------------------------------------------------------------------------------# 


# # Testing GMM

      # GMM_plot_diff_component_eval(DTR,LTR,DTE,LTE)
      # GMM_plot_diff_component_eval_PCA(DTR, LTR, DTE, LTE)
      # GMM_Tied_plot_diff_component_eval_PCA(DTR, LTR, DTE, LTE)
      # GMM_Diagonal_plot_diff_component_eval_znorm(DTR, LTR, DTE, LTE)
      # GMM_test_compare(DTR, LTR, DTE, LTE)


#----------------------------------------------------------------------------------------# 



