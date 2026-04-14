# -*- coding: ascii -*-

"""
Base module defining the interface for benchmark evaluation metrics.

This module provides the generic `MultipleRunMetric` base class, establishing
the standard evaluation protocol for change point detection algorithms
over a complete dataset or benchmark suite.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.detection_trace import DetectionTrace


class MultipleRunMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any], ResultT](ABC):
    """
    Abstract base class for all benchmark evaluation metrics.

    Provides a generic interface to evaluate a collection of detection traces
    against their corresponding labeled ground truth datasets.
    """

    @abstractmethod
    def evaluate(self, runs: Sequence[tuple[TraceT, ProviderT]]) -> ResultT:
        """
        Evaluate the detection traces against the provided labeled data.

        Parameters
        ----------
        runs : Sequence[tuple[TraceT, ProviderT]]
            A sequence of pairs containing the detection trace and the
            ground truth data provider.

        Returns
        -------
        ResultT
            The computed metric result over the entire benchmark.
        """
