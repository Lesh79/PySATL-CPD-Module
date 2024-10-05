"""
Module for implementation of Bayesian CPD algorithm.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm
from CPDShell.Core.algorithms.BayesianCPD.abstracts.idetector import IDetector
from CPDShell.Core.algorithms.BayesianCPD.abstracts.ihazard import IHazard
from CPDShell.Core.algorithms.BayesianCPD.abstracts.ilikelihood import ILikelihood
from CPDShell.Core.algorithms.BayesianCPD.abstracts.ilocalizer import ILocalizer


class BayesianAlgorithm(Algorithm):
    """
    The class implementing Bayesian change point detection algorithm. It uses likelihood and hazard functions to update
    Bayesian statistics and detector/localizer to detect/localize a change point. The algorithm consists of 3 repeating
    stages:
    1) Learning the likelihood's parameters;
    2) Evaluation Bayesian statistics;
    3) Processing a changepoint in case there's one.
    """

    def __init__(
        self, learning_steps: int, likelihood: ILikelihood, hazard: IHazard, detector: IDetector, localizer: ILocalizer
    ):
        """
        Initializes a new instance of Bayesian algorithm module with given customization.
        :param learning_steps: number of steps to learn likelihood's parameters.
        :param likelihood: likelihood function for the given model.
        :param hazard: hazard function for the given model.
        :param detector: detector for change point detection from a run lengths distribution at the moment.
        :param localizer: localizer for change point localization from a run lengths distribution at the moment.
        """
        self._learning_steps = learning_steps

        self.__likelihood = likelihood
        self.__hazard = hazard

        self.__detector = detector
        self.__localizer = localizer

        self.__growth_probs = np.array([])
        self.__time = 0
        self.__gap_size = 0
        self.__pred_probs_are_zero = False

        self.__change_points: list[int] = []
        self.__change_points_count = 0

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """Finds change points in window.

        :param window: part of global data for finding change points.
        :return: the number of change points in the window.
        """
        self.__process_data(False, window)
        return self.__change_points_count

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """Finds coordinates of change points (localizes them) in window.

        :param window: part of global data for finding change points.
        :return: list of window change points.
        """
        self.__process_data(True, window)
        return self.__change_points.copy()

    def __process_data(self, with_localization: bool, window: Iterable[float | np.float64]) -> None:
        """
        Processes a window of data to detect/localize all change points depending on working mode.
        :param with_localization: boolean flag representing whether function needs to localize a change point.
        :param window: part of global data for change points analysis.
        """
        sample = list(window)
        sample_size = len(sample)
        if sample_size == 0:
            return

        self.__prepare(sample_size)

        while self.__time + self._learning_steps < sample_size:
            self.__learning_stage(sample)

            self.__bayesian_stage(sample)

            if self.__time < sample_size - 1:
                self.__process_change_point(sample_size, with_localization)

    def __learning_stage(self, sample: list[float | np.float64]) -> None:
        """
        Performs a likelihood's parameter learning stage.
        :param sample: an overall sample the model working with.
        """
        self.__likelihood.learn(sample[self.__time : self.__time + self._learning_steps])
        self.__shift_time(self._learning_steps - 1)

    def __bayesian_stage(self, sample: list[float | np.float64]) -> None:
        """
        Performs a Bayesian statistics (run lengths distribution) evaluating stage.
        :param sample: an overall sample the model working with.
        """
        sample_size = len(sample)
        self.__gap_size = 0

        while self.__bayesian_condition(sample_size):
            assert self.__time >= 0
            observation = sample[self.__time]
            self.__shift_time(1)
            self.__gap_size += 1
            assert self.__gap_size > 0

            self.__bayesian_update(observation)

    def __process_change_point(self, sample_size: int, with_localization: bool) -> None:
        """
        Updating a change points data depending on working mode (with or without localization) based on run lengths
        distribution.
        :param sample_size: an overall size of the sample.
        :param with_localization: boolean flag representing whether function needs to localize a change point.
        """
        self.__change_points_count += 1
        if with_localization:
            if self.__pred_probs_are_zero:
                self.__change_points.append(self.__time)
            else:
                run_length = self.__localizer.localize(self.__growth_probs[: self.__gap_size])
                assert 0 <= run_length <= sample_size

                change_point = self.__time - run_length + 1
                self.__change_points.append(change_point)
                self.__shift_time(-run_length + 1)

        self.__clear(sample_size)

    def __bayesian_condition(self, sample_size: int) -> bool:
        """
        A helper function checking conditions (time boundaries, zero predictive probabilities case,
        existence of detected change point) to continue Bayesian statistics evaluation.
        :param sample_size: an overall size of the sample.
        """
        return (
            self.__time < sample_size - 1
            and not self.__pred_probs_are_zero
            and not self.__pred_probs_are_zero
            and not self.__detector.detect(self.__growth_probs[: self.__gap_size + 1])
        )

    def __bayesian_update(self, observation: float | np.float64) -> None:
        """
        Performs a Bayesian update of statistics (run lengths distribution).
        :param observation: an observation from a sample.
        """
        assert not self.__pred_probs_are_zero

        # 3. Evaluate predictive probabilities for all run lengths and it's parameters inside likelihood functions.
        predictive_probs = self.__likelihood.predict(observation)

        # Assuming that an abrupt change in all predictive probabilities to zero corresponds to a change point at this
        # moment.
        if np.count_nonzero(predictive_probs) == 0:
            self.__pred_probs_are_zero = True
            return

        # 4. Evaluate the hazard function for the gap.
        hazard_val = np.array(self.__hazard.hazard(np.array(range(self.__gap_size))))

        # Evaluate the changepoint probability at *this* step (NB: generally it can be found later, with some delay).
        changepoint_prob = np.sum(self.__growth_probs[0 : self.__gap_size] * predictive_probs * hazard_val)

        # Evaluate growth probabilities, shifting them down and to the right,
        # scaled by (1 - hazard function value) and prediction probabilities.
        self.__growth_probs[1 : self.__gap_size + 1] = (
            self.__growth_probs[0 : self.__gap_size] * predictive_probs * (1.0 - hazard_val)
        )

        # 5. Add CP probability.
        self.__growth_probs[0] = changepoint_prob

        # 6. Evaluate evidence for growth probabilities renormalization.
        evidence = np.sum(self.__growth_probs[0 : self.__gap_size + 2])

        # 7. Renormalize growth probabilities.
        assert evidence > 0.0
        self.__growth_probs[0 : self.__gap_size + 2] = self.__growth_probs[0 : self.__gap_size + 2] / evidence

        assert np.all(np.logical_and(self.__growth_probs >= 0.0, self.__growth_probs <= 1.0))

        # 8. Update parameters of likelihood function for every possible run length (typically appends new values).
        self.__likelihood.update(observation)

    def __shift_time(self, shift: int) -> None:
        """
        A helper function performing a time shift (adding a shift to current time).
        :param shift: a time shift to add to the current time.
        """
        self.__time += shift

    def __prepare(self, sample_size: int) -> None:
        """
        Clear algorithm's state (including change points and time related information) before data processing.
        :param sample_size: an overall size of the sample.
        """
        self.__time = 0
        self.__gap_size = 0

        self.__change_points = []
        self.__change_points_count = 0

        self.__clear(sample_size)

    def __clear(self, sample_size: int) -> None:
        """
        A helper function clearing a state of the model after a change point occurs.
        :param sample_size: an overall size of the sample.
        """
        self.__pred_probs_are_zero = False
        self.__likelihood.clear()
        self.__detector.clear()

        new_size = sample_size - self.__time
        self.__growth_probs = np.zeros(new_size)

        if new_size > 0:
            self.__growth_probs[0] = 1.0
