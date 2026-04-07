"""
Module for abstract base classes for SSA CPD algorithm.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.algorithms.ssa.abstracts.idecomposition import SVD, IDecomposition
from pysatl_cpd.core.algorithms.ssa.abstracts.idetector import IDetectorSSA
from pysatl_cpd.core.algorithms.ssa.abstracts.iembedding import IEmbedding
from pysatl_cpd.core.algorithms.ssa.abstracts.igrouping import IGrouping

__all__ = ["SVD", "IDecomposition", "IDetectorSSA", "IEmbedding", "IGrouping"]
