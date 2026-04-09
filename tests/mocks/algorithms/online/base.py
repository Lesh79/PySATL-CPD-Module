# -*- coding: ascii -*-

"""
Base mock classes for online algorithm testing.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from dataclasses import dataclass

from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.core.typedefs import Number


@dataclass(frozen=True, kw_only=True)
class MockAlgorithmState[T](OnlineAlgorithmState):
    """
    Mock algorithm state for testing.

    Parameters
    ----------
    is_in_learning_period : bool, default=False
        Indicates whether algorithm is in learning period.
    last_observation : Any | None, default=None
        Last observation passed to process().
    process_count : int, default=0
        Number of observations processed.
    """

    last_observation: T | None = None
    process_count: int = 0


@dataclass(frozen=True, kw_only=True)
class MockAlgorithmConfiguration(OnlineAlgorithmConfiguration):
    """
    Mock algorithm configuration for testing.

    Parameters
    ----------
    learning_period_size : int, default=0
        Number of initial observations for learning period.
    return_sequence : tuple[Number, ...]
        Sequence of return values for process() calls.
    """

    return_sequence: tuple[Number, ...]
