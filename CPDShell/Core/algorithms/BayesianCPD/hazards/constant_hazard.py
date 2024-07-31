"""
Module for implementation of Bayesian CPD algorithm constant hazard function corresponding to an exponential
distribution.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np

from CPDShell.Core.algorithms.BayesianCPD.abstracts.ihazard import IHazard


class ConstantHazard(IHazard):
    """
    A constant hazard function, corresponding to an exponential distribution with a given rate.
    """

    def __init__(self, rate: float):
        """
        Initializes the constant hazard function with a given rate of an underlying exponential distribution.
        :param rate: rate of an underlying exponential distribution.
        """
        self.__rate = rate

    def hazard(self, run_lengths: np.ndarray) -> np.ndarray:
        """
        Calculates the constant hazard function.
        :param run_lengths: run lengths at the time.
        :return: hazard function's values for given run lengths.
        """
        return np.ones(len(run_lengths)) / self.__rate
