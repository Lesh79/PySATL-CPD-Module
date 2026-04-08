# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.detection_trace import DetectionTrace


class RunMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any], ResultT](ABC):
    @abstractmethod
    def evaluate(self, runs: Sequence[tuple[TraceT, ProviderT]]) -> ResultT:
        pass
