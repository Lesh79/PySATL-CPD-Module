# -*- coding: ascii -*-
"""
Time series visualizer interface.

This module defines the abstract base class for visualizers that render
time series data with change point annotations.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from typing import Any, Self

from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer
from pysatl_cpd.core.data_providers import DataProvider


class ITimeseriesVisualizer[DataProviderT: DataProvider[Any]](IVisualizer, ABC):
    """
    Abstract base class for time series visualizers.

    Visualizers of this type render the original time series data,
    optionally with ground truth change points, detected change points,
    and annotation of learning and skip periods.

    Type Parameters
    ---------------
    DataProviderT : DataProvider
        The data provider type bound by DataProvider, containing the
        observations to be visualized.

    Notes
    -----
    The type parameter DataProviderT is bound to DataProvider to ensure
    that any concrete implementation works with valid data providers.
    """

    @abstractmethod
    def set_data_provider(self, data_provider: DataProviderT) -> Self:
        """
        Set the data provider containing the time series observations.

        Parameters
        ----------
        data_provider : DataProviderT
            Data provider that yields observations sequentially.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        raise NotImplementedError
