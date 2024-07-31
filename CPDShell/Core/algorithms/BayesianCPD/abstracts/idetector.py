"""
Module for Bayesian CPD algorithm detector's abstract base class.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod

import numpy as np


class IDetector(ABC):
    """
    Abstract base class for detectors that detect a change point with given growth probabilities for run lengths.
    """

    @abstractmethod
    def detect(self, growth_probs: np.ndarray) -> bool:
        """
        Checks whether a changepoint occurred with given growth probabilities at the time.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: boolean indicating whether a changepoint occurred
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        Clears the detector's state.
        :return:
        """
        ...
