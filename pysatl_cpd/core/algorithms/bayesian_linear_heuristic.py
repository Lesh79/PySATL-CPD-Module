"""
Module for chanhe point detection online algorithm, based on Bayesian online algorithm with heuristic, turning it into
an algorithm with linear time complexity with a cost of some information loss.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import copy
from typing import Optional

import numpy as np
from numpy import typing as npt

from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class BayesianLinearHeuristic(OnlineAlgorithm):
    """An online change point detection algorithm, based on changing the main Bayesian online algorithm instance to the
    duplicating time after some time. Note: this heuristic, however makes an algorithm linear on big time series, leads
    to some information loss, which may lead to some unstability in output's correctness."""

    def __init__(self, algorithm: BayesianOnline, time_before_duplicate_start: int, duplicate_preparation_time: int):
        """Initializes the Bayesian change point detection algorithm with linear time-complexity heuristc..

        :param algorithm: The base algorithm instance to use for detection/localization.
        :param time_before_duplicate_start: Time steps before starting duplicate algorithm's preparation (training
        and Bayesian modeling).
        :param duplicate_preparation_time: Time steps required to prepare (train and perform Bayesian modeling) the
        duplicating algorithm.
        :raises ValueError: If time constraints are not satisfied.
        :return:
        """
        if not (time_before_duplicate_start > duplicate_preparation_time > 0):
            raise ValueError(
                "time_before_duplicate_start must be greater than duplicate_preparation_time, which must be positive"
            )

        self.__original_algorithm = copy.deepcopy(algorithm)
        self.__time_before_duplicate_start = time_before_duplicate_start
        self.__duplicate_preparation_time = duplicate_preparation_time
        self.__main_algorithm = copy.deepcopy(algorithm)
        self.__duplicating_algorithm: Optional[BayesianOnline] = None
        self.__time = 0
        self.__last_algorithm_start_time = 0

    @property
    def __work_time(self) -> int:
        """
        Returns the number of steps since the last algorithm start.
        :return: the number of steps since the last algorithm start.
        """
        return self.__time - self.__last_algorithm_start_time

    def _handle_duplicate_preparation(
        self, observation: np.float64 | npt.NDArray[np.float64], method_name: str
    ) -> None:
        """
        Manages the creation and training, Bayesian modeling of the duplicating algorithm.

        :param observation: a new observation from a time series.
        :param method_name: the method to call on the duplicating algorithm ('detect'/'localize').
        :return:
        """
        work_time = self.__work_time
        stage_end = self.__time_before_duplicate_start + self.__duplicate_preparation_time

        # Start initializing duplicating algorithm
        if work_time == self.__time_before_duplicate_start:
            self.__duplicating_algorithm = copy.deepcopy(self.__original_algorithm)

        # Train the duplicating algorithm amd perform a Bayesian modeling during preparation period
        elif self.__time_before_duplicate_start < work_time < stage_end:
            if self.__duplicating_algorithm is not None:
                getattr(self.__duplicating_algorithm, method_name)(observation)

        # Switch to the prepared duplicating algorithm
        elif work_time == stage_end:
            assert self.__duplicating_algorithm is not None, "Duplicating algorithm must be initialized"
            self.__main_algorithm = copy.deepcopy(self.__duplicating_algorithm)
            self.__duplicating_algorithm = None
            self.__last_algorithm_start_time = self.__time - self.__duplicate_preparation_time

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Processes an observation and returns whether a change point was detected by a main algorithm.
        :param observation: a new observation from a time series. Note: only univariate data is supported for now.
        :return: whether a change point was detected by a main algorithm.
        """
        if isinstance(observation, np.ndarray):
            raise TypeError("Multivariate observations are not supported")
        assert self.__main_algorithm is not None, "Main algorithm must be initialized"

        # Run main detection
        if self.__main_algorithm.detect(observation):
            self.__last_algorithm_start_time = self.__time
            self.__duplicating_algorithm = None
            self.__time += 1
            return True

        # Manage duplicating algorithm training
        self._handle_duplicate_preparation(observation, "detect")
        self.__time += 1
        return False

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Processes an observation and returns the change point if localized by the main algorithm.
        :param observation: a new observation from a time series. Note: only univariate data is supported for now.
        :return: a change point, if it was localized, None otherwise.
        """
        if isinstance(observation, np.ndarray):
            raise TypeError("Multivariate observations are not supported")
        assert self.__main_algorithm is not None, "Main algorithm must be initialized"

        # Run main localization
        if (result := self.__main_algorithm.localize(observation)) is not None:
            change_point = self.__last_algorithm_start_time + result
            self.__last_algorithm_start_time = change_point
            self.__duplicating_algorithm = None
            self.__time += 1
            return change_point

        # Manage duplicating algorithm training
        self._handle_duplicate_preparation(observation, "localize")
        self.__time += 1
        return None
