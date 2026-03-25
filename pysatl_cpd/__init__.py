"""
PySATL CPD - Change Point Detection Library

A comprehensive library for change-point detection in time series data,
supporting both online and offline algorithms with type safety and
comprehensive testing.
"""

__version__ = "0.1.0"
__author__ = "PySATL contributors"
__license__ = "MIT"

from pysatl_cpd.core import data_providers
from pysatl_cpd.core.online import online_cpd_solver, online_detection_trace

__all__ = [
    "data_providers",
    "online_cpd_solver",
    "online_detection_trace",
]
