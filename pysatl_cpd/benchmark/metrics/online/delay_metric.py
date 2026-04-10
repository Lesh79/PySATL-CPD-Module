# -*- coding: ascii -*-

"""
Module for computing aggregated detection delays over a dataset.

Evaluates how quickly an online algorithm reacts to true change points
across multiple runs, providing Mean and Median delay metrics. Missed
detections are penalized with a maximum allowable delay.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from itertools import chain
from typing import Any

from numpy import mean, median

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.online.delay_metric import DelayMetric
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class MeanDelayMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, list[int], float]
):
    """
    Computes the mean detection delay over the entire dataset.

    Aggregates delay values from all runs into a single list and calculates
    the mean. Missed true change points contribute the `max_delay` penalty.

    Parameters
    ----------
    max_delay : int
        The maximum allowable delay window. Used as a penalty for missed
        change points and as the fallback result if no delays exist.
    """

    def __init__(self, max_delay: int) -> None:
        self.__base_metric = DelayMetric[TraceT, ProviderT](max_delay)
        self.__max_delay = max_delay

    @property
    def base_metric(self) -> DelayMetric[TraceT, ProviderT]:
        return self.__base_metric

    def aggregate(self, results: Sequence[list[int]]) -> float:
        """
        Aggregate delay lists by flattening and computing the mean.

        Parameters
        ----------
        results : Sequence[list[int]]
            A sequence of delay lists collected from each dataset run.

        Returns
        -------
        float
            Mean of all delays, or `max_delay` if the dataset contains
            no true change points.
        """

        all_delays = list(chain.from_iterable(results))
        if not all_delays:
            return float(self.__max_delay)

        return float(mean(all_delays))


class MedianDelayMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, list[int], float]
):
    """
    Computes the median detection delay over the entire dataset.

    Aggregates delay values from all runs into a single list and calculates
    the median. Missed true change points contribute the `max_delay` penalty.

    Parameters
    ----------
    max_delay : int
        The maximum allowable delay window. Used as a penalty for missed
        change points and as the fallback result if no delays exist.
    """

    def __init__(self, max_delay: int) -> None:
        self.__base_metric = DelayMetric[TraceT, ProviderT](max_delay)
        self.__max_delay = max_delay

    @property
    def base_metric(self) -> DelayMetric[TraceT, ProviderT]:
        return self.__base_metric

    def aggregate(self, results: Sequence[list[int]]) -> float:
        """
        Aggregate delay lists by flattening and computing the median.

        Parameters
        ----------
        results : Sequence[list[int]]
            A sequence of delay lists collected from each dataset run.

        Returns
        -------
        float
            Median of all delays, or `max_delay` if the dataset contains
            no true change points.
        """

        all_delays = list(chain.from_iterable(results))
        if not all_delays:
            return float(self.__max_delay)

        return float(median(all_delays))
