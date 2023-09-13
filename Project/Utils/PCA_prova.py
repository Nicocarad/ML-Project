import numpy as np


def compute_covariance_matrix(d: np.ndarray) -> np.ndarray:
    mu = compute_mean(d)
    return (1 / d.shape[1]) * np.dot((d - mu), (d - mu).T)

def compute_mean(d: np.ndarray) -> np.ndarray:
    return v_col(d.mean(1))


def v_col(x: np.ndarray) -> np.ndarray:
    return x.reshape((x.size, 1))


class Pca:

    def __init__(self, m: int):
        self.M = m
        self.eigen_value = None
        self.DTR = None
        self.DTE = None



    def process(self,DTR,DTE) -> tuple:
        self.DTR = DTR
        self.DTE = DTE
        # Evaluate the mean and center DTR and DTE over this mean
        mean = compute_mean(self.DTR)
        self.DTR = self.DTR - mean
        if self.DTE is not None:
            self.DTE = self.DTE - mean
        c = compute_covariance_matrix(self.DTR)
        # Evaluate the eigenvectors and directions
        self.eigen_value, u = np.linalg.eigh(c)
        p = u[:, ::-1][:, 0:self.M]
        # Compute the projections over the m directions p
        self.DTR = np.dot(p.T, self.DTR)
        if self.DTE is not None:
            self.DTE = np.dot(p.T, self.DTE)
        return self.DTR, self.DTE

    def __str__(self):
        return "PCA_M_" + str(self.M)