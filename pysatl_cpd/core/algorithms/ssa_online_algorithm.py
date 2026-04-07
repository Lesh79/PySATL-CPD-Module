"""
Module for SSA online change point detection algorithm.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm
from pysatl_cpd.core.algorithms.ssa.abstracts import IDetectorSSA
from pysatl_cpd.core.algorithms.ssa.ssa import SSA


class SSAOnline(OnlineAlgorithm):
    """
    Class for SSA online change point detection algorithm.
    """

    def __init__(
        self,
        ssa: SSA,
        detector: IDetectorSSA,
        N: int,
        L: int | None = None,
        p: int | None = None,
        Q: int | None = None,
    ) -> None:
        self.__ssa = ssa
        self.__detector = detector
        self.__N = N
        self.__L = int(N / 2) if L is None else L
        self.__p = N if p is None else p
        self.__Q = 1 if Q is None else Q
        self.__q = self.__p + self.__Q

        self.__buffer: list[np.float64] = []
        self.__current_time = 0
        self.__required_len = max(N, self.__q + self.__L - 1)

        self.__is_ready = False
        self.__was_changed = False
        self.__change_point: int | None = None

    def clear(self) -> None:
        """
        Clears the state of the algorithm's instance.
        :return:
        """
        self.__buffer = []
        self.__current_time = 0

        self.__is_ready = False
        self.__was_changed = False
        self.__change_point = None

    def __update_buffer(self) -> None:
        """
        Updates the buffer after detecting a breakpoint.
        :return:
        """
        self.__buffer = self.__buffer[self.__p :]
        if len(self.__buffer) == self.__required_len:
            self.__buffer.pop(0)
        self.__is_ready = False

    def __detect_breakpoint(self, with_localization: bool) -> None:
        """
        Checks for a breakoint and determines when.
        :param with_localization: whether the method was called for localization of a change point.
        :return:
        """
        training_data: npt.NDArray[np.float64] = np.array(self.__buffer[: self.__N])
        test_data = self.__buffer[self.__p :]

        subspace = self.__ssa.subspace(training_data, self.__L)
        detection = self.__detector.detect(subspace, test_data)

        if detection:
            self.__was_changed = True
            if with_localization:
                self.__change_point = self.__current_time - (
                    len(self.__buffer) - self.__p
                )

            self.__update_buffer()
            return

        self.__buffer.pop(0)

    def __process_point(self, observation: np.float64, with_localization: bool) -> None:
        """
        Universal method for processing of another observation of a time series.
        :param observation: new observation of a time series.
        :param with_localization: whether the method was called for localization of a change point.
        :return:
        """
        self.__buffer.append(observation)
        self.__current_time += 1

        if not self.__is_ready:
            if len(self.__buffer) != self.__required_len:
                return

            self.__is_ready = True

        self.__detect_breakpoint(with_localization)

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Performs a change point detection after processing another observation of a time series.
        :param observation: new observation of a time series. Note: multivariate time series aren't supported for now.
        :return: whether a change point was detected after processing the new observation.
        """
        if isinstance(observation, np.ndarray):
            raise TypeError("Multivariate observations are not supported")
        self.__process_point(observation, False)
        result = self.__was_changed
        self.__was_changed = False
        return result

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> int | None:
        """
        Performs a change point localization after processing another observation of a time series.
        :param observation: new observation of a time series.
        :return: absolute location of a change point, acquired after processing the new observation,
        or None if there wasn't any.
        """
        if isinstance(observation, np.ndarray):
            raise TypeError("Multivariate observations are not supported")
        self.__process_point(observation, True)
        result = self.__change_point
        self.__was_changed = False
        self.__change_point = None
        return result
