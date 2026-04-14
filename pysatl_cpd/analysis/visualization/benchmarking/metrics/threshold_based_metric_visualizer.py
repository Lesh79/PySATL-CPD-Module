# -*- coding: ascii -*-
"""
Threshold-based metric visualizer.
"""

from __future__ import annotations

from typing import Literal, Self

import plotly.graph_objects as go
import pandas as pd

from pysatl_cpd.analysis.visualization.benchmarking.abstracts import Axes, MetricVisualizer
from pysatl_cpd.analysis.visualization.typedefs import Figure, GoFigure, PltFigure

PrecisionMode = Literal["default", "monotonic", "both"]
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


class ThresholdBasedMetricVisualizer(MetricVisualizer):
    """
    Draw metrics as functions of threshold.
    """

    def __init__(
        self,
        *,
        y_metrics: list[str],
        title: str | None = None,
        ylabel: str | None = None,
        precision_mode: PrecisionMode = "default",
        style_map: dict[str, dict[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self._y_metrics = y_metrics
        self._title = title or "Threshold-based metrics"
        self._ylabel = ylabel or "Value"
        self._precision_mode = precision_mode
        self._style_map = style_map or {}

    @property
    def requirements(self) -> list[str]:
        columns = ["threshold", *self._y_metrics]
        return list(dict.fromkeys(columns))

    def set_benchmark_table(self, benchmark_table: pd.DataFrame) -> Self:
        return super().set_benchmark_table(benchmark_table)

    def draw(self, figure: Figure, axes: Axes) -> Figure:
        self._validate_required_columns()
        if self._precision_mode not in {"default", "monotonic", "both"}:
            raise ValueError(f"Unsupported precision mode: {self._precision_mode}")
        if isinstance(axes, tuple):
            return self._draw_plotly(figure, axes)
        return self._draw_matplotlib(figure, axes)

    # def _recalculate_stats(self, table: pd.DataFrame) -> pd.DataFrame:
    #     result = table.sort_values(by="threshold", ascending=True).copy()
    #     if "precision" in result.columns:
    #         result["precision"] = result["precision"].cummax()
    #     if "f1" in result.columns and "precision" in result.columns and "recall" in result.columns:
    #         precision = result["precision"]
    #         recall = result["recall"]
    #         result["f1"] = (2 * precision * recall) / (precision + recall).replace(0, pd.NA)
    #     return result
    # TODO: Add new column for monotonic precision outside of the visualizer by oveloading
    def _iter_mode_tables(self, table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        table = table.sort_values(by="threshold", ascending=True)
        mode_tables: list[tuple[str, pd.DataFrame]] = []
        if self._precision_mode in {"default", "both"}:
            mode_tables.append(("default", table))
        if self._precision_mode in {"monotonic", "both"}:
            mode_tables.append(("monotonic", self._recalculate_stats(table)))
        return mode_tables

    def _to_plotly_color(self, color: str | None) -> str | None:
        if color is None:
            return None
        return TAB_COLOR_TO_HEX.get(color, color)

    def _draw_matplotlib(self, figure: Figure, ax: Axes) -> PltFigure:
        table = self._require_table()
        mode_tables = self._iter_mode_tables(table)
        for mode, mode_table in mode_tables:
            mode_suffix = "" if mode == "default" else " (monotonic)"
            for metric in self._y_metrics:
                style = self._style_map.get(metric, {})
                ax.plot(
                    mode_table["threshold"],
                    mode_table[metric],
                    label=f"{metric}{mode_suffix}",
                    linestyle=style.get("linestyle", "--" if mode == "monotonic" else "-"),
                    color=style.get("color"),
                    linewidth=float(style.get("linewidth", 1)),
                )
        ax.set_title(self._title)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(self._ylabel)
        ax.grid(True, alpha=MPL_GRID_ALPHA)
        ax.legend(loc="best")
        return figure  # type: ignore[return-value]

    def _draw_plotly(self, figure: Figure, ax_pos: Axes) -> GoFigure:
        table = self._require_table()
        row, col = ax_pos  # type: ignore[misc]
        mode_tables = self._iter_mode_tables(table)
        for mode, mode_table in mode_tables:
            mode_suffix = "" if mode == "default" else " (monotonic)"
            for metric in self._y_metrics:
                style = self._style_map.get(metric, {})
                line_style = {
                    "dash": style.get("linestyle", "dash" if mode == "monotonic" else "solid"),
                    "width": float(style.get("linewidth", 1)),
                }
                color = self._to_plotly_color(style.get("color"))
                if color is not None:
                    line_style["color"] = color
                figure.add_trace(
                    go.Scatter(
                        x=mode_table["threshold"],
                        y=mode_table[metric],
                        mode="lines",
                        name=f"{metric}{mode_suffix}",
                        line=line_style,
                    ),
                    row=row,
                    col=col,
                )
        figure.update_xaxes(
            title_text="Threshold",
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
