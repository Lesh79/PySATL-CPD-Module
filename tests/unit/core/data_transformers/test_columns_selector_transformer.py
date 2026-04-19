# -*- coding: ascii -*-

"""
Tests for Data Transformers.

Covers IDataTransformer base class properties and ColumnsSelectorTransformer logic.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.core.data_providers.numpy_data_provider import (
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
)
from pysatl_cpd.core.data_transformers.columns_selector_transformer import (
    ColumnsSelectorTransformer,
)


class TestColumnsSelectorTransformer:
    """Tests for ColumnsSelectorTransformer logic and naming."""

    def test_name_single_column(self) -> None:
        """Transformer name should be formatted as 'Col_X' for a single int."""
        transformer = ColumnsSelectorTransformer(columns=2)
        assert transformer.name == "Col_2"

    def test_name_multiple_columns(self) -> None:
        """Transformer name should be formatted as 'Cols_X_Y' for a list of ints."""
        transformer = ColumnsSelectorTransformer(columns=[0, 2, 3])
        assert transformer.name == "Cols_0_2_3"

    def test_transform_int_to_univariate(self) -> None:
        """Selecting a single int column should yield a Univariate provider."""
        data: np.ndarray = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        provider = NDArrayMultivariateProvider(data=data, name="test_data")
        transformer = ColumnsSelectorTransformer(columns=1)

        result_provider = transformer.transform(provider)

        # Check type and name
        assert isinstance(result_provider, NDArrayUnivariateProvider)
        assert result_provider.name == "test_data_Col_1"

        # Check extracted data (column index 1 -> [2.0, 5.0, 8.0])
        result_data: list[float] = list(result_provider)
        np.testing.assert_array_equal(result_data, [2.0, 5.0, 8.0])

    def test_transform_list_to_multivariate(self) -> None:
        """Selecting a list of columns should yield a Multivariate provider."""
        data: np.ndarray = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
        provider = NDArrayMultivariateProvider(data=data, name="multidataset")
        transformer = ColumnsSelectorTransformer(columns=[0, 3])

        result_provider = transformer.transform(provider)

        # Check type and name
        assert isinstance(result_provider, NDArrayMultivariateProvider)
        assert result_provider.name == "multidataset_Cols_0_3"

        # Check extracted data (columns 0 and 3)
        result_data: list[np.ndarray] = list(result_provider)
        expected_data: list[np.ndarray] = [
            np.array([1.0, 4.0]),
            np.array([5.0, 8.0]),
        ]

        assert len(result_data) == 2
        np.testing.assert_array_equal(result_data[0], expected_data[0])
        np.testing.assert_array_equal(result_data[1], expected_data[1])

    def test_transform_raises_value_error_on_1d_data(self) -> None:
        """Attempting to select columns from 1D data should raise ValueError."""
        data: np.ndarray = np.array([1.0, 2.0, 3.0])
        provider = NDArrayUnivariateProvider(data=data, name="1d_data")
        transformer = ColumnsSelectorTransformer(columns=0)

        expected_msg = "ColumnsSelectorTransformer expects 2D data, got 1D data from provider '1d_data'."
        with pytest.raises(ValueError, match=expected_msg):
            transformer.transform(provider)  # type: ignore[arg-type]

    def test_transform_raises_index_error_on_out_of_bounds(self) -> None:
        """Passing an out-of-bounds column index should propagate an IndexError from NumPy."""
        data: np.ndarray = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        provider = NDArrayMultivariateProvider(data=data, name="data")

        # Array only has columns 0 and 1, index 5 is out of bounds
        transformer = ColumnsSelectorTransformer(columns=5)

        with pytest.raises(IndexError):
            transformer.transform(provider)
