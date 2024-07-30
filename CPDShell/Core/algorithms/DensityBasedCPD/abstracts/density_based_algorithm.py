import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class DensityBasedAlgorithm(Algorithm):
    @staticmethod
    def _kernel_density_estimation(observation, bandwidth):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(observation)
        return kde

    @staticmethod
    def _calculate_weights(test_value, reverence_value, bandwidth, regularization_coef, objective_function):
        test_density = DensityBasedAlgorithm._kernel_density_estimation(test_value, bandwidth)
        reference_density = DensityBasedAlgorithm._kernel_density_estimation(reverence_value, bandwidth)

        def obj(alpha):
            density_ratio = np.exp(test_density.score_samples(test_value) - reference_density.score_samples(test_value) - alpha)
            return objective_function(density_ratio, alpha)

        res = minimize(obj, np.zeros(len(test_value)), method="L-BFGS-B")
        alpha = res.x
        density_ratio = np.exp(test_density.score_samples(test_value) - reference_density.score_samples(test_value) - alpha)
        return density_ratio / np.mean(density_ratio)
