"""
Module for Bayesian CPD algorithm localizer's abstract base class.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod

import numpy as np


class ILocalizer(ABC):
    """
    Abstract base class for localizers that localize a change point with given growth probabilities for run lengths.
    """

    @abstractmethod
    def localize(self, growth_probs: np.ndarray) -> int:
        """
        Localizes a change point with given growth probabilities for run lengths.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: run length corresponding with a change point.
        """
        ...
