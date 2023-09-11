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

    def train(self, DTR, LTR, DTE, eff_prior, covariance_func):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR

        num_classes = numpy.unique(self.LTR).size

        gmm = [
            lbg_algorithm(
                self.iterations,
                self.DTR[:, self.LTR == classes],
                [[1, *mean_and_covariance(self.DTR[:, self.LTR == classes])] ],
                self.alpha,
                self.psi,
                covariance_func,
            )
            for classes in range(num_classes)
        ]

        self.gmm = gmm

    def compute_scores(self):
        self.scores = compute_gmm_scores(self.DTE, self.LTR, self.gmm)


class GMM_Tied(GMM):
    def train(self, DTR, LTR, DTE, eff_prior):
        super().train(DTR, LTR, DTE, eff_prior, tied_cov)


class GMM_Diagonal(GMM):
    def train(self, DTR, LTR, DTE, eff_prior):
        super().train(DTR, LTR, DTE, eff_prior, diagonal_cov)


class GMM_TiedDiagonal(GMM):
    def train(self, DTR, LTR, DTE, eff_prior):
        super().train(DTR, LTR, DTE, eff_prior, TiedDiagonal_cov)

