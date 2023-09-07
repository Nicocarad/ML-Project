from Utils.Kfold import *
from Utils.DCF import min_DCF
from Training.Gaussian.gaussian import *


def train_gaussianClassifiers(D, L):
    classifier_list = [
        # (LogGaussianClassifier, 0.5),
        # (LogGaussianClassifier, 0.1),
        # (LogGaussianClassifier, 0.9),
        # (NaiveBayes_GaussianClassifier, 0.5),
        # (NaiveBayes_GaussianClassifier, 0.1),
        # (NaiveBayes_GaussianClassifier, 0.9),
        # (TiedGaussianClassifier, 0.5),
        # (TiedGaussianClassifier, 0.1),
        # (TiedGaussianClassifier, 0.9),
        (Tied_NaiveBayes_GaussianClassifier, 0.5),
        (Tied_NaiveBayes_GaussianClassifier, 0.1),
        (Tied_NaiveBayes_GaussianClassifier, 0.9)
    ]

    for classifier, value in classifier_list:
        SPost, Label = kfold(classifier, 5, D, L, value)
        res = min_DCF(value, 1, 1, Label, SPost)
        print(f"Min DCF ({value}): ", res)
        
        
        
        
def train_Gaussians_PCA(D,L):
    



