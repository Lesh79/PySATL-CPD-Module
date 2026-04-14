# -*- coding: ascii -*-
"""
ARL-based metric visualizer.
"""

from __future__ import annotations

from typing import Self

import plotly.graph_objects as go
import pandas as pd

from pysatl_cpd.analysis.visualization.benchmarking.abstracts import Axes, MetricVisualizer
from pysatl_cpd.analysis.visualization.typedefs import Figure, GoFigure, PltFigure

PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"
MPL_GRID_ALPHA = 0.3

TAB_COLOR_TO_HEX = {
    "tab:blue": "#1f77b4",
    "tab:orange": "#ff7f0e",
    "tab:green": "#2ca02c",
    "tab:red": "#d62728",
    "tab:purple": "#9467bd",
    "tab:brown": "#8c564b",
    "tab:pink": "#e377c2",
    "tab:gray": "#7f7f7f",
    "tab:olive": "#bcbd22",
    "tab:cyan": "#17becf",
}


class ARLBasedMetricVisualizer(MetricVisualizer):
    """
    Draw metrics as functions of ARL.
    """

    def __init__(
        self,
        *,
        y_metrics: list[str],
        title: str = "ARL Curve",
        ylabel: str = "Values",
        style_map: dict[str, dict[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self._y_metrics = y_metrics
        self._title = title
        self._ylabel = ylabel
        self._style_map = style_map or {}

    @property
    def requirements(self) -> list[str]:
        columns = ["arl", *self._y_metrics]
        return list(dict.fromkeys(columns))

    def set_benchmark_table(self, benchmark_table: pd.DataFrame) -> Self:
        return super().set_benchmark_table(benchmark_table)

    def draw(self, figure: Figure, axes: Axes) -> Figure:
        self._validate_required_columns()
        if isinstance(axes, tuple):
            return self._draw_plotly(figure, axes)
        return self._draw_matplotlib(figure, axes)

    def _to_plotly_color(self, color: str | None) -> str | None:
        if color is None:
            return None
        return TAB_COLOR_TO_HEX.get(color, color)

    def _draw_matplotlib(self, figure: Figure, ax: Axes) -> PltFigure:
        table = self._require_table().sort_values(by="arl", ascending=True)
        for metric in self._y_metrics:
            style = self._style_map.get(metric, {})
            ax.plot(
                table["arl"],
                table[metric],
                label=metric,
                linestyle=style.get("linestyle", "-"),
                color=style.get("color"),
                linewidth=float(style.get("linewidth", 1)),
            )

        ax.set_title(self._title)
        ax.set_xlabel("ARL")
        ax.set_ylabel(self._ylabel)
        ax.grid(True, alpha=MPL_GRID_ALPHA)
        ax.legend(loc="best")
        return figure  # type: ignore[return-value]

    def _draw_plotly(self, figure: Figure, ax_pos: Axes) -> GoFigure:
        table = self._require_table().sort_values(by="arl", ascending=True)
        row, col = ax_pos  # type: ignore[misc]
        for metric in self._y_metrics:
            style = self._style_map.get(metric, {})
            line_style = {
                "dash": style.get("linestyle", "solid"),
                "width": float(style.get("linewidth", 1)),
            }
            color = self._to_plotly_color(style.get("color"))
            if color is not None:
                line_style["color"] = color
            figure.add_trace(
                go.Scatter(
                    x=table["arl"],
                    y=table[metric],
                    mode="lines",
                    name=metric,
                    line=line_style,
                ),
                row=row,
                col=col,
            )

        figure.update_xaxes(
            title_text="ARL",
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text=self._ylabel,
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        return figure  # type: ignore[return-value]
