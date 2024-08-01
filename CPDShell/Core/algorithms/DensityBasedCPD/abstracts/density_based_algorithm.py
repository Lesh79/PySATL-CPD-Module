from abc import abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
from scipy.optimize import minimize

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class DensityBasedAlgorithm(Algorithm):
    @staticmethod
    def _kernel_density_estimation(observation: np.ndarray, bandwidth: float) -> np.ndarray:
        """Perform kernel density estimation on the given observations without fitting a model.

        Args:
            observation (np.ndarray): the data points for which to estimate the density.
            bandwidth (float): the bandwidth parameter for the kernel density estimation.

        Returns:
            np.ndarray: estimated density values for the observations.
        """
        n = len(observation)
        x_grid = np.linspace(np.min(observation) - 3 * bandwidth, np.max(observation) + 3 * bandwidth, 1000)
        kde_values = np.zeros_like(x_grid)
        for x in observation:
            kde_values += np.exp(-0.5 * ((x_grid - x) / bandwidth) ** 2)

        kde_values /= n * bandwidth * np.sqrt(2 * np.pi)
        return kde_values

    @staticmethod
    def _calculate_weights(
        self,
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
        test_density = self._kernel_density_estimation(test_value, bandwidth)
        reference_density = self._kernel_density_estimation(reference_value, bandwidth)

        def objective_function_wrapper(alpha: np.ndarray) -> float:
            """Wrapper for the objective function to calculate the density ratio.

            Args:
                alpha (np.ndarray): relative parameter that controls the weighting between the numerator distribution
                and the denominator distribution in the density ratio estimation.

            Returns:
                float: the value of the objective function to minimize.
            """
            objective_density_ratio = np.exp(test_density - reference_density - alpha)
            return objective_function(objective_density_ratio, alpha)

        res = minimize(objective_function_wrapper, np.zeros(len(test_value)), method="L-BFGS-B")
        alpha = res.x
        objective_density_ratio = np.exp(test_density - reference_density - alpha)
        return objective_density_ratio / np.mean(objective_density_ratio)

    @abstractmethod
    def detect(self, window: Iterable[float]) -> int:
        # maybe rtype tuple[int]
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: list of right borders of window change points
        """
        raise NotImplementedError

    @abstractmethod
    def localize(self, window: Iterable[float]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        raise NotImplementedError
