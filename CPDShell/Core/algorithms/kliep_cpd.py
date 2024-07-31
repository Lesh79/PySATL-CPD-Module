from collections.abc import Iterable
from typing import List, Callable

import numpy as np

from CPDShell.Core.algorithms.DensityBasedCPD.abstracts.density_based_algorithm import DensityBasedAlgorithm


class KliepAlgorithm(DensityBasedAlgorithm):
    """Kullback-Leibler Importance Estimation Procedure (KLIEP) algorithm
    for change point detection.

    KLIEP estimates the density ratio between two distributions and uses
    the importance weights for detecting changes in the data distribution.
    """

    def __init__(self, bandwidth: float, regularization_coef: float, threshold: float = 1.1):
        """Initialize the KLIEP algorithm.

        Args:
            bandwidth (float): bandwidth parameter for density estimation.
            regularization_coef (float): regularization parameter.
            threshold (float, optional): threshold for detecting change points.
            Defaults to 1.1.
        """
        self.bandwidth = bandwidth
        self.regularization_coef = regularization_coef
        self.threshold = threshold

    def _loss_function(self, weights: np.ndarray, alpha: np.ndarray) -> float:
        """Loss function for KLIEP.

        Args:
            weights (np.ndarray): weights for the density estimation.
            alpha (np.ndarray): coefficients for the density ratio.

        Returns:
            float: the computed loss value.
        """
        return -np.mean(weights) + self.regularization_coef * np.sum(alpha**2)

    def detect(self, window: Iterable[float]) -> int:
        """Detect the number of change points in the given data window
        using KLIEP.

        Args:
            window (Iterable[float]): the data window to detect change points.

        Returns:
            int: the number of detected change points.
        """
        weights = self._calculate_weights(
            data=window,
            reference=window,
            bandwidth=self.bandwidth,
            regularization_coef=self.regularization_coef,
            loss_function=self._loss_function,
        )

        return np.count_nonzero(weights > self.threshold)

    def localize(self, window: Iterable[float]) -> List[int]:
        """Localize the change points in the given data window using KLIEP.

        Args:
            window (Iterable[float]): the data window to localize
            change points.

        Returns:
            List[int]: the indices of the detected change points.
        """
        weights = self._calculate_weights(
            data=window,
            reference=window,
            bandwidth=self.bandwidth,
            regularization_coef=self.regularization_coef,
            loss_function=self._loss_function,
        )

        return np.where(weights > self.threshold)[0].tolist()

    def evaluate_detection_accuracy(self, true_change_points: List[int], detected_change_points: List[int]) -> dict:
        """Evaluate the accuracy of change point detection.

        Args:
            true_change_points (List[int]): list of true change point indices.
            detected_change_points (List[int]): list of detected change point indices.

        Returns:
            dict: a dictionary with evaluation metrics (precision, recall, F1 score).
        """
        true_positive = len(set(true_change_points) & set(detected_change_points))
        false_positive = len(set(detected_change_points) - set(true_change_points))
        false_negative = len(set(true_change_points) - set(detected_change_points))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative,
        }
