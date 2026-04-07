"""
Module for SSA CPD algorithm detector's abstract base class.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IDetectorSSA(ABC):
    """
    Abstract class for detectors that detect a change point.
    """

    @abstractmethod
    def detect(
        self, subspace: npt.NDArray[np.float64], test_data: list[np.float64]
    ) -> bool:
        """
        Checks whether a changepoint has occurred at the start of the test data.
        :param subspace: vectors of the training set subspace.
        :param test_data: test sample for breakpoint detection.
        :return: boolean indicating whether a changepoint occurred.
        """

    @abstractmethod
    def clean(self) -> None:
        """
        Clears the detector's state.
        """
