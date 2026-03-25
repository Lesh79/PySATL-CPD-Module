# -*- coding: ascii -*-

"""
Mock online algorithm implementations for testing.

This module provides a flexible mock implementation of OnlineAlgorithm
for testing change-point detection components.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from tests.mocks.algorithms.online.base import (
    MockAlgorithmConfiguration,
    MockAlgorithmState,
)
from tests.mocks.algorithms.online.error import MockErrorOnlineAlgorithm
from tests.mocks.algorithms.online.simple import MockOnlineAlgorithm

__all__ = ["MockAlgorithmState", "MockAlgorithmConfiguration", "MockOnlineAlgorithm", "MockErrorOnlineAlgorithm"]
