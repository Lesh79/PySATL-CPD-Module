"""
Base mock classes for online algorithm testing.
"""

from dataclasses import dataclass, field
from typing import Any

from pysatl_cpd.online.ionline_algorithm import (
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)


@dataclass(frozen=True, kw_only=True)
class MockAlgorithmState(OnlineAlgorithmState):
    """
    Mock algorithm state for testing.

    This state extends the base OnlineAlgorithmState with testing-specific
    fields for tracking algorithm behavior.

    Parameters
    ----------
    is_in_learning_period : bool, default=False
        Indicates whether algorithm is in learning period.
    process_count : int, default=0
        Number of observations processed.
    last_observation : Number | None, default=None
        Last observation passed to process().
    custom_data : dict[str, Any], default_factory=dict
        Custom data for testing state evolution.
    """

    process_count: int = 0
    last_observation: float | None = None
    custom_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MockAlgorithmConfiguration(OnlineAlgorithmConfiguration):
    """
    Mock algorithm configuration for testing.

    This configuration extends the base OnlineAlgorithmConfiguration with
    testing-specific parameters.

    Parameters
    ----------
    learning_period_size : int, default=0
        Number of initial observations for learning period.
    custom_param : int, default=0
        Custom integer parameter for testing.
    custom_string : str, default="default"
        Custom string parameter for testing.
    """

    custom_param: int = 0
    custom_string: str = "default"
