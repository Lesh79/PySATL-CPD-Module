"""
Module for implementation of Bayesian CPD algorithm gaussian (normal) likelihood function with unknown mean and
variance. It uses normal-inverse gamma distribution as a conjugate prior function and Student's t-distribution as
a predictive probability.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt
from scipy import stats

from CPDShell.Core.algorithms.BayesianCPD.abstracts.ilikelihood import ILikelihood


class GaussianUnknownMeanAndVariance(ILikelihood):
    """
    Likelihood for Gaussian (a.k.a. normal) distribution with unknown mean and variance estimated from normal-inverse
    gamma distribution as a conjugate prior. It uses 4 parameters, which priors are estimated from a learning sample and
    iteratively updated after an observation. Predictive probability is Student's t-distribution with posterior
    parameters.
    """

    def __init__(self):
        """
        Initializes model. There are no known parameters at this moment.
        """
        self.__mu_0 = None
        self.__k_0 = None
        self.__alpha_0 = None
        self.__beta_0 = None

        self.__mu_params = np.array([])
        self.__k_params = np.array([])
        self.__alpha_params = np.array([])
        self.__beta_params = np.array([])

    def learn(self, learning_sample: list[float | np.float64]) -> None:
        """
        Learns first prior parameters. Can be interpreted as mean was estimated from k_0 observations with sample mean
        mu_0, and precision was estimated from 2 * alpha observations with sample mean mu_0 and sum of squared
        deviations 2 * beta.
        :param learning_sample: a sample for parameter learning.
        """
        data = np.array(learning_sample)
        sample_size = data.shape[0]
        self.__mu_0 = data.mean()
        self.__beta_0 = ((data - self.__mu_0) ** 2).sum() / 2.0
        self.__k_0 = sample_size
        self.__alpha_0 = sample_size / 2.0

        self.__mu_params = np.array([self.__mu_0])
        self.__k_params = np.array([self.__k_0])
        self.__alpha_params = np.array([self.__alpha_0])
        self.__beta_params = np.array([self.__beta_0])

    def update(self, observation: float | np.float64) -> None:
        """
        Updates 4 parameters arrays of normal-inverse gamma conjugate prior, calculating posterior parameters.
        :param observation: an observation from a sample.
        """
        mu_divider = self.__k_params + 1.0
        assert np.count_nonzero(mu_divider) == mu_divider.shape[0]

        beta_divider = 2.0 * self.__k_params + 1.0
        assert np.count_nonzero(beta_divider) == beta_divider.shape[0]

        new_mu_params = np.append([self.__mu_0], (self.__mu_params * self.__k_params + observation) / mu_divider)
        new_k_params = np.append([self.__k_0], self.__k_params + 1.0)
        new_alpha_params = np.append([self.__alpha_0], self.__alpha_params + 0.5)
        new_beta_params = np.append(
            [self.__beta_0],
            self.__beta_params + self.__k_params * (observation - self.__mu_params) ** 2 / beta_divider,
        )

        self.__mu_params = new_mu_params
        self.__k_params = new_k_params
        self.__alpha_params = new_alpha_params
        self.__beta_params = new_beta_params

    def predict(self, observation: float | np.float64) -> npt.ArrayLike:
        """
        Returns predictive probabilities for a given observation based on posterior parameters. Predictive distribution
        is Student's t-distribution with 2 * alpha degrees of freedom.
        :param observation: an observation from a sample.
        :return: predictive probabilities for a given observation.
        """

        scales_divider = self.__alpha_params * self.__k_params
        assert np.count_nonzero(scales_divider) == scales_divider.shape[0]

        degrees_of_freedom = 2.0 * self.__alpha_params
        scales = (self.__beta_params * (self.__k_params + 1.0)) / scales_divider

        predictive_probabilities = stats.t.pdf(
            x=observation,
            df=degrees_of_freedom,
            loc=self.__mu_params,
            scale=scales,
        )

        return predictive_probabilities

    def clear(self) -> None:
        """
        Clears parameters of gaussian likelihood.
        """
        self.__mu_0 = None
        self.__k_0 = None
        self.__alpha_0 = None
        self.__beta_0 = None

        self.__mu_params = np.array([])
        self.__k_params = np.array([])
        self.__alpha_params = np.array([])
        self.__beta_params = np.array([])
