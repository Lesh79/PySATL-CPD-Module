# -*- coding: ascii -*-

"""
Mock pandas data provider for testing segment logic.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.data_providers.dataset import PandasLabeledDataProvider, SegmentFilter


class MockPandasLabeledDataProvider(PandasLabeledDataProvider):
    """
    Mock implementation of PandasLabeledDataProvider for testing segment slicing.

    Bypasses pandas DataFrame initialization entirely and returns pre-configured
    bisegments and indices when queried.
    """

    def __init__(self, name: str = "MockPandasProvider") -> None:
        self._name = name
        self.mock_bisegments: list[PandasLabeledDataProvider] = []
        self.mock_indexes: list[tuple[int, int, int]] = []

    @property
    def name(self) -> str:
        return self._name

    def query_bisegments(self, filter_fn: SegmentFilter | None = None) -> list[PandasLabeledDataProvider]:
        """Return pre-configured bisegments."""
        return self.mock_bisegments

    def query_bisegments_indexes(self, filter_fn: SegmentFilter | None = None) -> list[tuple[int, int, int]]:
        """Return pre-configured bisegment indices."""
        return self.mock_indexes
