from abc import abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class DensityBasedAlgorithm(Algorithm):
    @staticmethod
    def _kernel_density_estimation(observation: np.ndarray, bandwidth: float) -> KernelDensity:
        """Perform kernel density estimation on the given observations.

        Args:
            observation (np.ndarray): the data points for which to estimate the density.
            bandwidth (float): the bandwidth parameter for the kernel density estimation.

        Returns:
            KernelDensity: a fitted KernelDensity object.
        """
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(observation)
        return kde

    @staticmethod
    def _calculate_weights(
        test_value: np.ndarray,
        reference_value: np.ndarray,
        bandwidth: float,
        objective_function: Callable[[np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
        """Calculate the weights based on the density ratio between test and reference values.

        Args:
            test_value (np.ndarray): the test data points.
            reference_value (np.ndarray): the reference data points.
            bandwidth (float): the bandwidth parameter for the kernel density estimation.
            objective_function (Callable[[np.ndarray, np.ndarray], float]): the objective function to minimize.

        Returns:
            np.ndarray: the calculated density ratios normalized to their mean.
        """
        test_density = DensityBasedAlgorithm._kernel_density_estimation(test_value, bandwidth)
        reference_density = DensityBasedAlgorithm._kernel_density_estimation(reference_value, bandwidth)

        def objective_function_wrapper(alpha: np.ndarray) -> float:
            """Wrapper for the objective function to calculate the density ratio.

            Args:
                alpha (np.ndarray): relative parameter that controls the weighting between the numerator distribution
                and the denominator distribution in the density ratio estimation.

            Returns:
                float: the value of the objective function to minimize.
            """
            objective_density_ratio = np.exp(
                test_density.score_samples(test_value) - reference_density.score_samples(test_value) - alpha
            )
            return objective_function(objective_density_ratio, alpha)

        res = minimize(objective_function_wrapper, np.zeros(len(test_value)), method="L-BFGS-B")
        alpha = res.x
        objective_density_ratio = np.exp(
            test_density.score_samples(test_value) - reference_density.score_samples(test_value) - alpha
        )
        return objective_density_ratio / np.mean(objective_density_ratio)

    @abstractmethod
    def detect(self, window: Iterable[float]) -> list[int]:
        # maybe rtype tuple[int]
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: list of right borders of window change points
        """
        ...

    @abstractmethod
    def localize(self, window: Iterable[float]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        ...
