"""
Module for implementation of Bayesian CPD algorithm localizer selecting the most probable run length.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np

from CPDShell.Core.algorithms.BayesianCPD.abstracts.ilocalizer import ILocalizer


class SimpleLocalizer(ILocalizer):
    """
    A localizer that localizes a change point corresponding with the most probable non-max run length.
    """

    def localize(self, growth_probs: np.ndarray) -> int:
        """
        Localizes a change point corresponding with the most probable non-max run length.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: the most probable non-max run length corresponding change point.
        """
        if len(growth_probs) == 0:
            return 0

        return int(growth_probs[0 : len(growth_probs) - 1].argmax())
