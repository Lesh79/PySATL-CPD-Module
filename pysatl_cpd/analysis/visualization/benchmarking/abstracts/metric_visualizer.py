# -*- coding: ascii -*-
"""
Abstract benchmark metric visualizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd

from pysatl_cpd.analysis.visualization.typedefs import Figure, GoAxes, PltAxes

type Axes = PltAxes | GoAxes

# TODO: Use IVisualizer interface
class MetricVisualizer(ABC):
    """
    Base class for benchmark metric visualizers.

    A metric visualizer draws a single subplot from a benchmark result table.
    """

    def __init__(self) -> None:
        self._benchmark_table: pd.DataFrame | None = None

    @property
    @abstractmethod
    def requirements(self) -> list[str]:
        """
        Return required columns for this visualizer.
        """
        raise NotImplementedError

    def set_benchmark_table(self, benchmark_table: pd.DataFrame) -> Self:
        """
        Store benchmark table.
        """
        self._benchmark_table = benchmark_table
        return self

    @abstractmethod
    # TODO: Change to AxMapping
    def draw(self, figure: Figure, axes: Axes) -> Figure:
        """
        Draw the metric on provided axes.
        """
        raise NotImplementedError

    def _require_table(self) -> pd.DataFrame:
        if self._benchmark_table is None:
            raise ValueError("Benchmark table is not set.")
        return self._benchmark_table

    def _validate_required_columns(self) -> None:
        table = self._require_table()
        missing = [column for column in self.requirements if column not in table.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
