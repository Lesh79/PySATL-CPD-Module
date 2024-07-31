"""
Module for implementation of Bayesian CPD algorithm gaussian (normal) likelihood function with mean and standard
deviation learning.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import numpy.typing as npt
from scipy import stats

from CPDShell.Core.algorithms.BayesianCPD.abstracts.ilikelihood import ILikelihood


class GaussianLikelihood(ILikelihood):
    """
    Likelihood for Gaussian (a.k.a. normal) distribution, parametrized by mean and standard deviation.
    """

    def __init__(self):
        """
        Initializes the GaussianLikelihood, parametrized by mean and standard deviation (without any concrete values).
        """
        self.__means = np.array([])
        self.__standard_deviations = np.array([])

        self.__sample_sum = 0.0
        self.__squared_sample_sum = 0.0
        self.__gap_size = 0

    def __update_parameters_lists(self) -> None:
        """
        Updates the parameters lists based on accumulated sums, assuming we have at least 2 observations.
        """
        assert self.__gap_size > 1
        new_mean = self.__sample_sum / self.__gap_size
        variance = (self.__squared_sample_sum - (self.__sample_sum**2.0) / self.__gap_size) / (self.__gap_size - 1)
        assert variance > 0.0
        assert len(self.__means) == len(self.__standard_deviations)

        new_standard_deviation = np.sqrt(variance)

        self.__means = np.append(self.__means, new_mean)
        self.__standard_deviations = np.append(self.__standard_deviations, new_standard_deviation)

    def learn(self, learning_sample: list[float | np.float64]) -> None:
        """
        Learns first mean and stander deviations from a given sample.
        :param learning_sample: a sample for parameter learning.
        :return:
        """
        assert len(self.__means) == len(self.__standard_deviations) == 0
        assert self.__gap_size == 0

        self.__sample_sum += sum(learning_sample)
        for observation in learning_sample:
            self.__squared_sample_sum += observation**2.0

        self.__gap_size = len(learning_sample)
        # self.__squared_sample_sum += sum(learning_sample ** 2.)

        self.__update_parameters_lists()

    def update(self, observation: float | np.float64) -> None:
        """
        Updates the means and standard deviations lists according to the given observation.
        :param observation: an observation from a sample.
        :return:
        """
        self.__sample_sum += observation
        self.__squared_sample_sum += observation**2
        self.__gap_size += 1

        self.__update_parameters_lists()

    def predict(self, observation: float | np.float64) -> npt.ArrayLike:
        """
        Returns predictive probabilities for a given observation based on stored means and standard deviations.
        :param observation: an observation from a sample.
        :return: predictive probabilities for a given observation.
        """
        return stats.norm(self.__means, self.__standard_deviations).pdf(observation)

    def clear(self):
        """
        Clears parameters of gaussian likelihood.
        :return:
        """
        self.__means = []
        self.__standard_deviations = []

        self.__sample_sum = 0.0
        self.__squared_sample_sum = 0.0
        self.__gap_size = 0
