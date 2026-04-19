# -*- coding: ascii -*-

"""
Tests for Data Transformers.

Covers IDataTransformer base class properties and ColumnsSelectorTransformer logic.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from tests.mocks.core.data_transformers.simple import DummyTransformer


class TestIDataTransformer:
    """Tests for the abstract IDataTransformer base class default behaviors."""

    def test_default_name_is_class_name(self) -> None:
        """The default name property should return the class name."""
        transformer = DummyTransformer()
        assert transformer.name == "DummyTransformer"

    def test_default_hash_is_hash_of_name(self) -> None:
        """The default hash should be equal to the hash of the transformer's name."""
        transformer = DummyTransformer()
        assert hash(transformer) == hash("DummyTransformer")
