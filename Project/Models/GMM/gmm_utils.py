import numpy
import scipy
from Utils.utils import *


def mean_and_covariance(data_matrix):
    N = data_matrix.shape[1]
    mu = mcol(data_matrix.mean(1))
    DC = data_matrix - mu
    C = numpy.dot(DC, DC.T) / N

    return mu, C


def logpdf_GAU_ND_fast(X, mu, C):
    X_c = X - mu
    M = X.shape[0]
    const = -0.5 * M * numpy.log(2 * numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (X_c * numpy.dot(L, X_c)).sum(0)

    return const - 0.5 * logdet - 0.5 * v


def logpdf_gmm(X, gmm):
    s = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(X.shape[1]):
        for idx, component in enumerate(gmm):
            s[idx, i] = logpdf_GAU_ND_fast(
                X[:, i : i + 1], component[1], component[2]
            ) + numpy.log(component[0])
    return scipy.special.logsumexp(s, axis=0)


def LBG(iterations, X, gmm_start, alpha, psi, covariance_func=None):
    if covariance_func is not None:
        gmm_start = covariance_func(gmm_start, [X.shape[1]], X.shape[1])

    for i in range(len(gmm_start)):
        covNew = gmm_start[i][2]
        U, s, _ = numpy.linalg.svd(covNew)
        s[s < psi] = psi
        gmm_start[i] = (gmm_start[i][0], gmm_start[i][1], numpy.dot(U, mcol(s) * U.T))
    gmm_start = EM(X, gmm_start, psi, covariance_func)

    for i in range(iterations):
        gmm_new = []
        for g in gmm_start:
            start_w = g[0]
            start_mu = g[1]
            start_sigma = g[2]

            U, s, _ = numpy.linalg.svd(start_sigma)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha

            new_w = start_w / 2
            new_mu = start_mu
            new_sigma = start_sigma

            gmm_new.append((new_w, new_mu + d, new_sigma))
            gmm_new.append((new_w, new_mu - d, new_sigma))
        gmm_start = EM(X, gmm_new, psi, covariance_func)
    return gmm_start


def EM(X, gmm, psi, covariance_func=None):
    llr_1 = None
    while True:
        num_components = len(gmm)

        logS = numpy.zeros((num_components, X.shape[1]))

        # START E-STEP
        for idx in range(num_components):
            logS[idx, :] = logpdf_GAU_ND_fast(X, gmm[idx][1], gmm[idx][2]) + numpy.log(
                gmm[idx][0]
            )
        logSMarginal = scipy.special.logsumexp(
            logS, axis=0
        )  # compute marginal densities
        SPost = numpy.exp(logS - logSMarginal)

        # END E-STEP

        # START M-STEP

        Z_vec = numpy.zeros(num_components)

        gmm_new = []
        for idx in range(num_components):
            gamma = SPost[idx, :]

            # update model parameters
            Z = gamma.sum()
            F = (mrow(gamma) * X).sum(1)
            S = numpy.dot(X, (mrow(gamma) * X).T)

            Z_vec[idx] = Z

            # new parameters
            mu = mcol(F / Z)
            sigma = S / Z - numpy.dot(mu, mu.T)
            w = Z / X.shape[1]

            gmm_new.append((w, mu, sigma))
        # END M-STEP

        # if Tied / Diagonal / Tied-Diagonal
        if covariance_func is not None:
            gmm_new = covariance_func(gmm_new, Z_vec, X.shape[1])

        # Constraining the eigenvalues
        for i in range(num_components):
            covNew = gmm_new[i][2]
            U, s, _ = numpy.linalg.svd(covNew)
            s[s < psi] = psi
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], numpy.dot(U, mcol(s) * U.T))
        gmm = gmm_new

        # stop criterion
        llr_0 = llr_1
        llr_1 = numpy.mean(logSMarginal)
        if llr_0 is not None and llr_1 - llr_0 < 1e-6:
            break

    return gmm


def gmm_scores(D, L, gmm):
    unique_labels = numpy.unique(L)
    num_classes = unique_labels.size
    num_features = D.shape[1]

    scores = numpy.zeros((num_classes, num_features))
    for i, label in enumerate(unique_labels):
        class_scores = logpdf_gmm(D, gmm[label])
        scores[i] = numpy.exp(class_scores)

    llr = numpy.log(scores[1] / scores[0])

    return llr


def tied_cov(gmm, z_vec, n):
    num_components = len(gmm)
    new_sigma = numpy.zeros_like(gmm[0][2])

    for g in range(num_components):
        new_sigma += gmm[g][2] * z_vec[g]

    new_sigma /= n

    updated_gmm = [(component[0], component[1], new_sigma) for component in gmm]

    return updated_gmm



def diagonal_cov(gmm, _z_vec, _n):
    return [(g[0], g[1], numpy.diag(numpy.diag(g[2]))) for g in gmm]



def TiedDiagonal_cov(gmm, z_vec, n):
    tied_diagonal_gmm = diagonal_cov(tied_cov(gmm, z_vec, n), z_vec, n)
    return tied_diagonal_gmm
