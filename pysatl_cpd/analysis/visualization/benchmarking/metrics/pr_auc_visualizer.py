# -*- coding: ascii -*-
"""
Precision-Recall AUC visualizer.
"""

from __future__ import annotations

from typing import Self

import numpy as np
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


class PrAucVisualizer(MetricVisualizer):
    """
    Draw precision-recall curve and PR-AUC value.
    """

    def __init__(self, *, label: str = "PR-AUC", color: str = "tab:green", linestyle: str = "-") -> None:
        super().__init__()
        self._label = label
        self._color = color
        self._linestyle = linestyle

    @property
    def requirements(self) -> list[str]:
        return ["recall", "precision"]

    def set_benchmark_table(self, benchmark_table: pd.DataFrame) -> Self:
        return super().set_benchmark_table(benchmark_table)

    def draw(self, figure: Figure, axes: Axes) -> Figure:
        self._validate_required_columns()
        if isinstance(axes, tuple):
            return self._draw_plotly(figure, axes)
        return self._draw_matplotlib(figure, axes)

    # TODO: Move to self._benchmark_table property
    def _prepare_pr_data(self, table: pd.DataFrame) -> pd.DataFrame:
        pr_data = (
            table[["recall", "precision"]]
            .sort_values(by=["recall", "precision"], ascending=[True, False])
            .drop_duplicates(subset=["recall"], keep="first")
        )
        # NOTE: Need to check if this is correct
        boundary_points = pd.DataFrame([{"recall": 0.0, "precision": 1.0}, {"recall": 1.0, "precision": 0.0}])
        return pd.concat([pr_data, boundary_points], ignore_index=True).sort_values(by="recall")

    # TODO: MOve outside of the visualizer. Pass to label
    def _compute_auc(self, pr_data: pd.DataFrame) -> float:
        return float(np.trapezoid(pr_data["precision"], pr_data["recall"]))

    # def _to_plotly_color(self, color: str) -> str:
    #     return TAB_COLOR_TO_HEX.get(color, color)

    # TODO: TypeDict for options. Add special setter for style map.
    def _draw_matplotlib(self, figure: Figure, ax: Axes) -> PltFigure:
        table = self._require_table()
        pr_data = self._prepare_pr_data(table)
        auc_score = self._compute_auc(pr_data)

        ax.plot(
            pr_data["recall"],
            pr_data["precision"],
            color=self._color,
            linestyle=self._linestyle,
            marker="o",
            label=f"{self._label} (AUC = {auc_score:.3f})",
        )
        ax.set_title("PR-AUC")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=MPL_GRID_ALPHA)
        ax.legend(loc="best")
        return figure  # type: ignore[return-value]

    def _draw_plotly(self, figure: Figure, ax_pos: Axes) -> GoFigure:
        table = self._require_table()
        row, col = ax_pos  # type: ignore[misc]
        pr_data = self._prepare_pr_data(table)
        auc_score = self._compute_auc(pr_data)

        figure.add_trace(
            go.Scatter(
                x=pr_data["recall"],
                y=pr_data["precision"],
                mode="lines+markers",
                name=f"{self._label} (AUC = {auc_score:.3f})",
                line={
                    "color": self._to_plotly_color(self._color),
                    "dash": "solid" if self._linestyle == "-" else "dash",
                },
            ),
            row=row,
            col=col,
        )
        figure.update_xaxes(
            title_text="Recall",
            range=[0.0, 1.05],
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text="Precision",
            range=[0.0, 1.05],
            showgrid=True,
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_layout(showlegend=True)
        return figure  # type: ignore[return-value]
