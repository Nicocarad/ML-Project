import numpy 
from scipy.optimize import fmin_l_bfgs_b










class Logistic_Regression:
    def __init__(self, l):
        self.DTR = 0
        self.LTR = 0
        self.DTE = 0
        self.l = l
        self.eff_prior = 0
        self.w = None
        self.b = None
        self.scores = 0

    def train(self, DTR, LTR, DTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.eff_prior = eff_prior
        
        x, _, _ = fmin_l_bfgs_b(
            self.log_reg_obj, numpy.zeros(self.DTR.shape[0] + 1), approx_grad=True, factr=1
        )

        self.w, self.b = x[:-1], x[-1]

    def log_reg_obj(self, v):
        w, b = v[:-1], v[-1]
        
        priors = numpy.array([self.eff_prior, 1 - self.eff_prior])
        
        normalizer = (self.l / 2) * numpy.linalg.norm(w) ** 2
        
        loss_funct = 0
        unique_classes = numpy.unique(self.LTR)

        for i in unique_classes:
            LTR_mask = self.LTR == i
            const_class = priors[i] / numpy.sum(LTR_mask)

            DTR_filtered = self.DTR[:, LTR_mask]
            z_filtered = 2 * self.LTR[LTR_mask] - 1

            inner_product = numpy.dot(w.T, DTR_filtered) + b
            exp_term = numpy.logaddexp(0, -z_filtered * inner_product)

            loss_funct += const_class * numpy.sum(exp_term)

        return normalizer + loss_funct

    def compute_scores(self):
        self.scores = numpy.dot(self.w.T, self.DTE) + self.b

