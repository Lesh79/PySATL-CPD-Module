# -*- coding: ascii -*-
"""
Dummy state visualizer implementation.

This module provides a placeholder visualizer for algorithm state evolution
that does nothing, useful for testing or when state visualization is not needed.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Self

from pysatl_cpd.analysis.visualization.online.states.ionline_state_visualizer import IOnlineStateVisualizer
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    DrawOption,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState


class DummyStateVisualizer[StateT: OnlineAlgorithmState](IOnlineStateVisualizer[StateT]):
    """
    Dummy state visualizer that performs no actual rendering.

    This visualizer implements the IOnlineStateEvolutionVisualizer interface
    but does nothing when drawn. It declares an empty set of axes, making it
    suitable for use as a placeholder when state visualization is not required
    or for testing purposes.

    Parameters
    ----------
    backend : DrawBackend
        Plotting backend to use (inherited from IVisualizer).
    """

    def __init__(self, backend: DrawBackend) -> None:
        """
        Initialize the dummy state visualizer.

        Parameters
        ----------
        backend : DrawBackend
            Plotting backend to use for rendering.
        """
        super().__init__(backend)
        self._states: Sequence[StateT | None] = []

    @property
    def axes(self) -> set[str]:
        """
        Declare the subplot names required by this visualizer.

        Returns
        -------
        set[str]
            Empty set indicating that this visualizer does not require any axes.
        """
        return set()

    def set_states(self, states: Sequence[StateT | None], **draw_options: DrawOption) -> Self:
        """
        Set the sequence of algorithm states.

        This implementation stores the states but does not use them for rendering.

        Parameters
        ----------
        states : Sequence[OnlineAlgorithmState | None]
            Sequence of algorithm state snapshots for each observation step.
        **draw_options : DrawOptions
            Additional backend-specific drawing options (ignored).

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._states = states
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """
        Draw using Matplotlib backend.

        This implementation does nothing and returns the figure unchanged.

        Parameters
        ----------
        figure : PltFigure
            Matplotlib figure to draw onto.
        axes : AxMapping
            Mapping from subplot names to their positions.

        Returns
        -------
        PltFigure
            Unmodified figure object.
        """
        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """
        Draw using Plotly backend.

        This implementation does nothing and returns the figure unchanged.

        Parameters
        ----------
        figure : GoFigure
            Plotly figure to draw onto.
        axes : AxMapping
            Mapping from subplot names to their positions.

        Returns
        -------
        GoFigure
            Unmodified figure object.
        """
        return figure
