# -*- coding: ascii -*-
"""
Module for computing aggregated metrics over specific dataset transitions (bisegments).
"""

__author__ = "Your Name"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any, cast

from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.core.data_providers.dataset import PandasLabeledDataProvider, SegmentFilter
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class SegmentAggregationMetric[TraceT: OnlineDetectionTrace[Any], ResultInT, ResultOutT](
    MultipleRunMetric[TraceT, PandasLabeledDataProvider, dict[str, ResultOutT]]
):
    """
    Evaluates an aggregation metric exclusively on specific transition types (bisegments).

    This metric slices both the input data providers and their corresponding
    detection traces based on user-provided transition filters. It then groups
    these slices by transition type and computes the underlying base metric for
    each group independently.

    Parameters
    ----------
    base_agg_metric : AggregationMetric[TraceT, PandasLabeledDataProvider, ResultInT, ResultOutT]
        The underlying metric to compute (e.g., F1Metric, MeanDelayMetric) for each group.
    transition_filters : dict[str, SegmentFilter]
        A mapping where keys are human-readable transition names (e.g., 'A -> B')
        and values are callable predicates that filter bisegments.
    """

    def __init__(
        self,
        base_agg_metric: AggregationMetric[TraceT, PandasLabeledDataProvider, ResultInT, ResultOutT],
        transition_filters: dict[str, SegmentFilter],
    ) -> None:
        self._base_agg_metric = base_agg_metric
        self._transition_filters = transition_filters

    @property
    def base_agg_metric(self) -> AggregationMetric[TraceT, PandasLabeledDataProvider, ResultInT, ResultOutT]:
        """
        Returns the underlying aggregation metric instance.
        """

        return self._base_agg_metric

    def evaluate(self, runs: Sequence[tuple[TraceT, PandasLabeledDataProvider]]) -> dict[str, ResultOutT]:
        """
        Evaluate the metric grouped by segment transitions.

        Parameters
        ----------
        runs : Sequence[tuple[TraceT, PandasLabeledDataProvider]]
            The full benchmark execution results.

        Returns
        -------
        dict[str, Rout]
            A dictionary mapping the transition name to the computed metric result.
            If a transition filter matches no segments, it is omitted from the output.
        """

        grouped_runs: dict[str, list[tuple[TraceT, PandasLabeledDataProvider]]] = {
            name: [] for name in self._transition_filters
        }

        for trace, provider in runs:
            for trans_name, filter_fn in self._transition_filters.items():
                sub_providers = provider.query_bisegments(filter_fn)
                sub_indices = provider.query_bisegments_indexes(filter_fn)

                for sub_prov, (g_start, _, g_end) in zip(sub_providers, sub_indices, strict=False):
                    sub_trace = cast(TraceT, trace.slice(g_start, g_end))
                    grouped_runs[trans_name].append((sub_trace, sub_prov))

        results: dict[str, ResultOutT] = {}
        for trans_name, sub_runs in grouped_runs.items():
            if sub_runs:
                results[trans_name] = self._base_agg_metric.evaluate(sub_runs)

        return results
