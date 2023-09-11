import numpy
from Training.GMM.gmm_utils import *


class GMM:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.DTR = 0
        self.DTE = 0
        self.LTR = 0
        self.gmm = None
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.scores = 0

    def train(self, DTR, LTR, DTE, eff_prior):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            lbg_algorithm(
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
        self.scores = gmm_scores(self.DTE, self.LTR, self.gmm)


class GMM_Tied:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.DTR = 0
        self.DTE = 0
        self.LTR = 0
        self.gmm = None
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.scores = 0

    def train(self, DTR, LTR, DTE, eff_prior, covariance_func):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            lbg_algorithm(
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
        self.scores = gmm_scores(self.DTE, self.LTR, self.gmm)


class GMM_Diagonal:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.DTR = 0
        self.DTE = 0
        self.LTR = 0
        self.gmm = None
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.scores = 0

    def train(self, DTR, LTR, DTE, eff_prior, covariance_func):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            lbg_algorithm(
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
        self.scores = gmm_scores(self.DTE, self.LTR, self.gmm)


class GMM_TiedDiagonal:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.DTR = 0
        self.DTE = 0
        self.LTR = 0
        self.gmm = None
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.scores = 0

    def train(self, DTR, LTR, DTE, eff_prior, covariance_func):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            lbg_algorithm(
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
        self.scores = gmm_scores(self.DTE, self.LTR, self.gmm)
