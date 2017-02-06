import numpy as np
import math


class ComputerMarginalLikelihood(object):
    @staticmethod
    def compute_marginal_likelihood(K, Y):
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
        log = 0.5 * L.T * alpha - np.trace(L) - (K.shape[0] / 2) * math.log(2 * math.pi)
        return log


class ComputerMarginalLikelihoodFactory(object):
    @staticmethod
    def create_marginal_computer():
        return ComputerMarginalLikelihood()
