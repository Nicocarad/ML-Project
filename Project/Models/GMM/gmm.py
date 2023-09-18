import numpy
from Models.GMM.gmm_utils import *


class GMM:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR
        self.LTE = LTE

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            LBG(
                self.iterations,
                self.DTR[:, self.LTR == classes],
                [[1, *mean_and_covariance(self.DTR[:, self.LTR == classes])]],
                self.alpha,
                self.psi,
            )
            for classes in range(num_classes)
        ]

        self.gmm = gmm

    def compute_scores(self):
        self.scores = gmm_scores(self.DTE, self.LTE, self.gmm)


class GMM_Tied:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR
        self.LTE = LTE

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            LBG(
                self.iterations,
                self.DTR[:, self.LTR == classes],
                [[1, *mean_and_covariance(self.DTR[:, self.LTR == classes])]],
                self.alpha,
                self.psi,
                tied_cov,
            )
            for classes in range(num_classes)
        ]

        self.gmm = gmm

    def compute_scores(self):
        self.scores = gmm_scores(self.DTE, self.LTE, self.gmm)


class GMM_Diagonal:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR
        self.LTE = LTE

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            LBG(
                self.iterations,
                self.DTR[:, self.LTR == classes],
                [[1, *mean_and_covariance(self.DTR[:, self.LTR == classes])]],
                self.alpha,
                self.psi,
                diagonal_cov,
            )
            for classes in range(num_classes)
        ]

        self.gmm = gmm

    def compute_scores(self):
        self.scores = gmm_scores(self.DTE, self.LTE, self.gmm)


class GMM_TiedDiagonal:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR
        self.LTE = LTE

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            LBG(
                self.iterations,
                self.DTR[:, self.LTR == classes],
                [[1, *mean_and_covariance(self.DTR[:, self.LTR == classes])]],
                self.alpha,
                self.psi,
                TiedDiagonal_cov,
            )
            for classes in range(num_classes)
        ]

        self.gmm = gmm

    def compute_scores(self):
        self.scores = gmm_scores(self.DTE, self.LTE, self.gmm)
