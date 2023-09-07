from Utils.Kfold import *
from Utils.DCF import min_DCF
from Training.Gaussian.gaussian import *
from Utils.PCA import PCA


def train_gaussianClassifiers(D, L):
    classifier_list = [
        (LogGaussianClassifier, 0.5),
        (LogGaussianClassifier, 0.1),
        (LogGaussianClassifier, 0.9),
        (NaiveBayes_GaussianClassifier, 0.5),
        (NaiveBayes_GaussianClassifier, 0.1),
        (NaiveBayes_GaussianClassifier, 0.9),
        (TiedGaussianClassifier, 0.5),
        (TiedGaussianClassifier, 0.1),
        (TiedGaussianClassifier, 0.9),
        (Tied_NaiveBayes_GaussianClassifier, 0.5),
        (Tied_NaiveBayes_GaussianClassifier, 0.1),
        (Tied_NaiveBayes_GaussianClassifier, 0.9)
    ]
    
    m_list = [11, 10, 9, 8, 7]

    print("GAUSSIAN CLASSIFIERS NO PCA\n")
    for classifier, value in classifier_list:
        SPost, Label = kfold(classifier, 5, D, L, value)
        res = min_DCF(value, 1, 1, Label, SPost)
        classifier_name = classifier.__name__
        print(f"Min DCF ({classifier_name}, {value}): ", {round(res, 3)})
    
    
    print("GAUSSIAN CLASSIFIERS + PCA(7-8-9-10-11)\n")
    for m in m_list:
        D_pca = PCA(D, m)
        print(f"Value of m: {m}")   
        for classifier, value in classifier_list:
                SPost, Label = kfold(classifier, 5, D_pca, L, value)
                res = min_DCF(value, 1, 1, Label, SPost)
                classifier_name = classifier.__name__
                print(f"Min DCF ({classifier_name}, {value}): ", round(res, 3))
        
        
        
        



