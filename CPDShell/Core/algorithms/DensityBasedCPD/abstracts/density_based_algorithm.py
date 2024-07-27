import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class DensityBasedAlgorithm(Algorithm):
    @staticmethod
    def _kde(X, bandwidth):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)
        return kde

    @staticmethod
    def _calculate_weights(Xt, Xr, bandwidth, lam, objective_function):
        pt = DensityBasedAlgorithm._kde(Xt, bandwidth)
        pr = DensityBasedAlgorithm._kde(Xr, bandwidth)

        def obj(alpha):
            w = np.exp(pt.score_samples(Xt) - pr.score_samples(Xt) - alpha)
            return objective_function(w, alpha)

        res = minimize(obj, np.zeros(len(Xt)), method="L-BFGS-B")
        alpha = res.x
        w = np.exp(pt.score_samples(Xt) - pr.score_samples(Xt) - alpha)
        return w / np.mean(w)
