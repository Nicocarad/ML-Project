from Utils.load import load
from Features_Analysis.feature_analysis import *
from Training.Gaussian.train_gaussian import *
import numpy


if __name__ == '__main__':


# Dataset Info
    D1, L1 = load('Dataset/Train.txt')
    D2, L2 = load('Dataset/Test.txt')
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
    
    
# Train Gaussian Classifiers
train_gaussianClassifiers(D1,L1)
    
