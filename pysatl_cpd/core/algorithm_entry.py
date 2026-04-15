# -*- coding: ascii -*-
"""
Container for benchmark algorithm execution entries.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pysatl_cpd.core.data_transformers.idata_transformer import IDataTransformer
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm, OnlineAlgorithmConfiguration, OnlineAlgorithmState


@dataclass
class AlgorithmEntry[DataT, ConfigT: OnlineAlgorithmConfiguration, StateT: OnlineAlgorithmState]:
    """
    Groups an algorithm, its target thresholds, and an optional data transformer.

    This container simplifies benchmark configuration by coupling the detection
    algorithm with the specific preprocessing steps (transformer) required for
    the target datasets.

    Parameters
    ----------
    algorithm : OnlineAlgorithm
        The instantiated online change-point detection algorithm.
    thresholds : Sequence[float]
        A sequence of detection thresholds to evaluate.
    transformer : IDataTransformer | None, optional
        Data transformer to apply to the dataset before feeding it to the algorithm.
        If None, data is passed as-is. Default is None.
    """

    algorithm: OnlineAlgorithm[DataT, ConfigT, StateT]
    thresholds: Sequence[float]
    transformer: IDataTransformer[Any, Any] | None = None

    @property
    def full_name(self) -> str:
        """
        Combined name of the algorithm and transformer.

        Returns
        -------
        str
            Name formatted as '{AlgorithmName}_{TransformerName}' or just
            '{AlgorithmName}' if no transformer is used.
        """
        algo_name = self.algorithm.name
        if self.transformer is not None:
            return f"{algo_name}_{self.transformer.name}"
        return algo_name

    @property
    def full_hash(self) -> int:
        """
        Combined hash of the algorithm configuration and transformer.

        Used to uniquely identify this execution pipeline in the cache registry.

        Returns
        -------
        int
            Hash value representing the execution configuration.
        """
        base_hash = hash(self.algorithm.configuration)
        if self.transformer is not None:
            return hash((base_hash, hash(self.transformer)))
        return base_hash
