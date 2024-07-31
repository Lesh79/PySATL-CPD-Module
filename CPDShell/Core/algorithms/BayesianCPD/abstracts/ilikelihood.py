"""
Module for Bayesian CPD algorithm likelihood function's abstract base class.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod

import numpy.typing as npt


class ILikelihood(ABC):
    """
    Likelihood function's abstract base class.
    """

    @abstractmethod
    def learn(self, learning_sample: list[float]) -> None:
        """
        Learns first parameters of a likelihood function on a given sample.
        :param learning_sample: a sample for parameter learning.
        :return:
        """
        ...

    @abstractmethod
    def predict(self, observation: float) -> npt.ArrayLike:
        """
        Returns predictive probabilities for a given observation based on stored parameters.
        :param observation: an observation from a sample.
        :return: predictive probabilities for a given observation.
        """
        ...

    @abstractmethod
    def update(self, observation: float) -> None:
        """
        Updates parameters of a likelihood function according to the given observation.
        :param observation: an observation from a sample.
        :return:
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        Clears likelihood function's state.
        :return:
        """
        ...
