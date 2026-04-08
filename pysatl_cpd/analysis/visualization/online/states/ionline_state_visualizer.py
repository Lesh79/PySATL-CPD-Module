# -*- coding: ascii -*-
"""
Online algorithm state evolution visualizer interface.

This module defines the abstract base class for visualizers that render
the evolution of online algorithm state over time.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

from pysatl_cpd.analysis.visualization.abstracts import IVisualizer
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState


class IOnlineStateVisualizer[OnlineAlgorithmStateT: OnlineAlgorithmState](IVisualizer, ABC):
    """
    Abstract base class for online algorithm state evolution visualizers.

    Visualizers of this type render the evolution of algorithm state
    over time, showing how internal statistics change with each observation.

    Type Parameters
    ---------------
    OnlineAlgorithmStateT : OnlineAlgorithmState
        The algorithm state type bound by OnlineAlgorithmState. This allows
        visualizers to work with specific state implementations such as
        ShewhartControlChartState.

    Notes
    -----
    The type parameter is bound to OnlineAlgorithmState to ensure that
    any concrete implementation works with valid algorithm state objects.
    """

    @abstractmethod
    def set_states(
        self,
        states: Sequence[OnlineAlgorithmStateT | None],
    ) -> Self:
        """
        Set the sequence of algorithm states to visualize.

        Parameters
        ----------
        states : Sequence[OnlineAlgorithmStateT | None]
            Sequence of algorithm state snapshots for each observation step.
            None values indicate steps where state was not captured.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        raise NotImplementedError
