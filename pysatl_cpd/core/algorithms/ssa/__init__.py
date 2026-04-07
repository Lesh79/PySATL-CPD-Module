"""
Module for SSA CPD algorithm's customization blocks.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.algorithms.ssa.abstracts import (
    SVD,
    IDecomposition,
    IDetectorSSA,
    IEmbedding,
    IGrouping,
)
from pysatl_cpd.core.algorithms.ssa.decomposition import BasicSVD
from pysatl_cpd.core.algorithms.ssa.detectors import DistanceThreshold
from pysatl_cpd.core.algorithms.ssa.embedding import BasicEmbedding
from pysatl_cpd.core.algorithms.ssa.grouping import ConstantGrouping
from pysatl_cpd.core.algorithms.ssa.ssa import SSA

__all__ = [
    "SSA",
    "SVD",
    "BasicEmbedding",
    "BasicSVD",
    "ConstantGrouping",
    "DistanceThreshold",
    "IDecomposition",
    "IDetectorSSA",
    "IEmbedding",
    "IGrouping",
]
