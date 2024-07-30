"""
MIT License

Copyright (c) 2024 Alexey Tatyanenko

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

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

    def __update_parameters_lists(self):
        """
        Updates the parameters lists based on accumulated sums, assuming we have at least 2 observations.
        :return:
        """
        assert self.__gap_size > 1
        new_mean = self.__sample_sum / self.__gap_size
        variance = (self.__squared_sample_sum - (self.__sample_sum**2.0) / self.__gap_size) / (self.__gap_size - 1)
        assert variance > 0.0
        assert len(self.__means) == len(self.__standard_deviations)

        new_standard_deviation = np.sqrt(variance)

        self.__means = np.append(self.__means, new_mean)
        self.__standard_deviations = np.append(self.__standard_deviations, new_standard_deviation)

    def learn(self, learning_sample: list[float]) -> None:
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

    def update(self, observation: float) -> None:
        """
        Updates the means and standard deviations lists according to the given observation.
        :param observation: an observation from a sample.
        :return:
        """
        self.__sample_sum += observation
        self.__squared_sample_sum += observation**2
        self.__gap_size += 1

        self.__update_parameters_lists()

    def predict(self, observation: float) -> npt.ArrayLike:
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
