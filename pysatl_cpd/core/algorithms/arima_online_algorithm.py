"""
Module for implementation of Predict&Compare CPD algorithm.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.arima.models.arima import ArimaModel
from pysatl_cpd.core.algorithms.arima.stats_tests.cusum import CuSum
from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class ArimaCusumAlgorithm(OnlineAlgorithm):
    """
    An online change point detection algorithm that combines an ARIMA model
    with multiple CUSUM detectors to monitor the model's residuals.
    """

    def __init__(
        self,
        training_size: int,
        h_coefficient: float = 5.0,
        ema_alpha: float = 0.05,
        p: int = 0,
        d: int = 0,
        q: int = 0,
    ):
        """
        Initializes the ArimaCusumAlgorithm.
        :param training_size: the number of data points for initial training.
        :param h_coefficient: a multiplier for the statistic's standard deviation to set the CUSUM 'h' threshold.
        :param ema_alpha: the smoothing factor for the exponential moving average (EMA) of the residual variance.
        :param p: the order of the model for the autoregressive.
        :param d: the order of the model for the differences.
        :param q: the order of the model for the moving average components.
        """
        self.__arima_model: ArimaModel = ArimaModel()
        self.__order: Optional[tuple[int, int, int]] = (p, d, q)

        self.__training_buffer: list[np.float64] = []
        self.__training_size = training_size
        self.__is_training: bool = True

        self.__h_coefficient = np.float64(h_coefficient)
        self.__ema_alpha = ema_alpha
        self.__sigma2: Optional[np.float64] = None

        self.__current_time: int = 0

        self.__data_history: list[np.float64] = []
        self.__was_change_point: bool = False
        self.__change_point: Optional[int] = None

        self.__cusum_detectors: dict[str, CuSum] = {
            "mean": CuSum(),
            "variance": CuSum(),
            "skewness": CuSum(),
            "kurtosis": CuSum(),
        }
        self.__residual_moment_baselines: dict[str, np.float64] = {
            "mean": np.float64(0.0),
            "variance": np.float64(0.0),
            "skewness": np.float64(0.0),
            "kurtosis": np.float64(0.0),
        }

        self.__prev_residual_norm: np.float64 = np.float64(0.0)

    def clear(self) -> None:
        self.__arima_model.clear()

        self.__training_buffer = []
        self.__is_training = True
        self.__sigma2 = None

        self.__current_time = 0
        self.__data_history = []
        self.__was_change_point = False
        self.__change_point = None

        for _, detector in self.__cusum_detectors.items():
            detector.clear()

        for residual_moment_baseline in self.__residual_moment_baselines:
            self.__residual_moment_baselines[residual_moment_baseline] = np.float64(0.0)

    def __fit_model(self) -> None:
        """
        Fits the ARIMA model on the collected buffer and calibrates CUSUM detectors.
        :return:
        """
        assert len(self.__training_buffer) >= self.__training_size, (
            "Training buffer is smaller than required training size."
        )
        residuals = self.__arima_model.fit(self.__training_buffer, self.__order)
        assert residuals is not None and len(residuals) != 0, "ARIMA fit did not produce valid residuals."

        self.__sigma2 = np.maximum(np.var(residuals), 1e-6)
        self.__recalculate_cusum_thresholds(residuals)

        self.__training_buffer = []
        self.__is_training = False

    def __collect_training_data(self, observation: np.float64) -> None:
        """
        Appends an observation to the training buffer and fits the model when full.
        :param observation: a new data point.
        :return:
        """
        self.__training_buffer.append(observation)

        if len(self.__training_buffer) >= self.__training_size:
            self.__fit_model()

    def __update_cusum_detectors(self, residual: np.float64) -> None:
        """
        Updates all CUSUM detectors based on a new residual.
        :param residual: the residual from the ARIMA model.
        :return:
        """
        assert self.__sigma2 is not None, "Sigma squared must be initialized before updating detectors."

        sigma = np.maximum(np.sqrt(self.__sigma2), 1e-3)
        normalized_residual = residual / sigma

        # Calculate statistics for each moment, centered around their expected values
        stats = {
            "mean": normalized_residual - self.__residual_moment_baselines["mean"],
            "variance": ((normalized_residual**2) - 1) - self.__residual_moment_baselines["variance"],
            "skewness": (normalized_residual**3) - self.__residual_moment_baselines["skewness"],
            "kurtosis": ((normalized_residual**4) - 3) - self.__residual_moment_baselines["kurtosis"],
        }
        self.__prev_residual_norm = normalized_residual

        for name, detector in self.__cusum_detectors.items():
            detector.update(stats[name])

    def __check_for_change(self) -> dict[str, np.bool]:
        """
        Polls each CUSUM detector for a change signal.
        :return: a dictionary indicating which, if any, detector has fired.
        """
        return {name: detector.is_change_detected() for name, detector in self.__cusum_detectors.items()}

    def __recalculate_cusum_thresholds(self, training_residuals: npt.NDArray[np.float64]) -> None:
        """
        Recalculates and updates the thresholds for all CUSUM detectors.
        :param training_residuals: the residuals produced during the training phase.
        :return:
        """
        assert len(training_residuals) != 0, "Cannot calculate thresholds on empty residuals."

        train_sigma = np.std(training_residuals)
        train_sigma = np.maximum(train_sigma, 1e-6)

        normalized_residuals = training_residuals / train_sigma

        stats_arrays = {
            "mean": normalized_residuals,
            "variance": (normalized_residuals**2) - 1,
            "skewness": normalized_residuals**3,
            "kurtosis": (normalized_residuals**4) - 3,
        }

        # Update moment baselines
        for name, arr in stats_arrays.items():
            self.__residual_moment_baselines[name] = np.float64(np.mean(arr))

        # Set k and h thresholds for each detector
        for name, stats_array in stats_arrays.items():
            stat_std = np.std(stats_array)
            stat_std = np.maximum(stat_std, 1e-6)

            k = np.float64(0.5) * stat_std
            h = self.__h_coefficient * stat_std

            self.__cusum_detectors[name].update_thresholds(k=k, h=h)

    def __update_model_and_detectors(self, observation: np.float64) -> None:
        """
        Updates the model with a new observation and runs the detection logic.
        :param observation: a new data point.
        :return:
        """
        prediction = self.__arima_model.predict(1)[0]
        residual = observation - prediction

        # Update variance estimate using EMA
        assert self.__sigma2 is not None, "Model must be trained before updating (sigma2 is None)."
        self.__sigma2 = (1 - self.__ema_alpha) * self.__sigma2 + self.__ema_alpha * (residual**2)

        self.__update_cusum_detectors(residual)
        self.__arima_model.update([observation])

    def __process_point(self, observation: np.float64) -> None:
        """
        Universal method for processing of another observation of a time series.
        :param observation: new observation of a time series.
        :return:
        """
        self.__current_time += 1
        self.__data_history.append(observation)

        if self.__is_training:
            self.__collect_training_data(observation)
        else:
            self.__update_model_and_detectors(observation)
            detections = self.__check_for_change()

            if any(detections.values()):
                self.__handle_change_point(detections)
                return

    def __handle_change_point(self, detections: dict[str, np.bool]) -> None:
        """
        Executes the logic to handle a detected change point, including localization
        and resetting the algorithm for retraining.
        :param detections: the dictionary of detector results.
        :return:
        """
        assert any(detections), "Handler called without any detection."
        self.__was_change_point = True

        # Localize the change point based on detector priority
        priority = ["mean", "variance", "skewness", "kurtosis"]
        for stat_name in priority:
            if detections.get(stat_name):
                self.__change_point = self.__current_time - self.__cusum_detectors[stat_name].get_last_reset_index()
                break

        self.__arima_model.clear()
        self.__is_training = True
        for detector in self.__cusum_detectors.values():
            detector.clear()

        self.__residual_moment_baselines = {
            "mean": np.float64(0.0),
            "variance": np.float64(0.0),
            "skewness": np.float64(0.0),
            "kurtosis": np.float64(0.0),
        }
        self.__prev_residual_norm = np.float64(0.0)
        self.__sigma2 = None

        # Keep data after the change point for retraining
        assert self.__change_point is not None
        time_after_cpd = self.__current_time - self.__change_point
        self.__data_history = self.__data_history[-time_after_cpd:] if time_after_cpd > 0 else []
        self.__training_buffer = []

        # Retrain on the data collected after the change point
        for observation in self.__data_history:
            if self.__is_training:
                self.__collect_training_data(observation)
            else:
                self.__update_model_and_detectors(observation)

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Function for finding change points in window

        :param observation: part of global data for finding change points
        :return: the number of change points in the window
        """
        if isinstance(observation, np.ndarray):
            raise TypeError("Multivariate observations are not supported")

        self.__process_point(np.float64(observation))
        result = self.__was_change_point
        self.__was_change_point = False
        return result

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Function for finding coordinates of change points in window

        :param observation: part of global data for finding change points
        :return: list of window change points
        """
        if isinstance(observation, np.ndarray):
            raise TypeError("Multivariate observations are not supported")

        self.__process_point(np.float64(observation))
        result = self.__change_point
        self.__was_change_point = False
        self.__change_point = None
        return result
