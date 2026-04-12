# -*- coding: ascii -*-
"""
Benchmark execution module for change-point detection algorithms.

This module provides the core components for running and caching performance
evaluations of online CPD algorithms across multiple datasets and threshold
configurations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import csv
import itertools
import math
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pysatl_cpd.core.data_providers.idata_provider import DataProvider
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


@dataclass
class BenchmarkRecord:
    """
    Metadata container for a single benchmark execution.

    This record uniquely identifies a benchmark run and stores the path
    to the cached trace file if disk dumping is enabled.

    Parameters
    ----------
    algorithm : str
        The string identifier or name of the online algorithm.
    configuration_hash : str
        A hash string representing the algorithm's configuration.
    data : str
        The identifier or name of the dataset.
    threshold : float
        The detection threshold used for this specific run.
    trace_path : str | None, default=None
        Absolute or relative path to the serialized detection trace file,
        if caching is enabled.
    """

    algorithm: str
    configuration_hash: int
    data: str
    threshold: float
    trace_path: str | None = None

    @property
    def key(self) -> tuple[str, int, str, float]:
        """
        Get the unique composite key for this benchmark run.

        Returns
        -------
        tuple[str, int, str, float]
            A tuple containing (algorithm, configuration_hash, data, threshold)
            used for identifying the record in the registry.
        """
        return (self.algorithm, self.configuration_hash, self.data, self.threshold)


class BenchmarkExecutor[DataT]:
    """
    Orchestrator for executing change-point detection benchmarks.

    Evaluates a set of algorithms across multiple data providers and thresholds
    using a provided online solver. Supports a caching mechanism via disk dumping
    to prevent redundant calculations on subsequent runs.

    Parameters
    ----------
    algorithms : list[tuple[OnlineAlgorithm[Any, Any, Any], Sequence[float]]]
        A list of tuples, where each tuple contains an instantiated online
        algorithm and a sequence of thresholds to test it against.
    providers : list[DataProvider[DataT]]
        A list of data providers to be fed into the algorithms.
    solver : OnlineCpdSolver
        The solver instance responsible for iterating over the data providers
        and running the algorithmic logic.
    dump_dir : str | Path | None, optional
        Directory path where the benchmark registry (CSV) and serialized traces
        (Pickle files) should be stored. If None, caching is disabled.
    """

    def __init__(
        self,
        algorithms: list[tuple[OnlineAlgorithm[Any, Any, Any], Sequence[float]]],
        providers: list[DataProvider[DataT]],
        solver: OnlineCpdSolver,
        dump_dir: str | Path | None = None,
    ) -> None:
        self.__algorithms = algorithms
        self.__providers = providers
        self.__solver = solver
        self.__dump_dir = Path(dump_dir) if dump_dir is not None else None

    def execute(self) -> list[tuple[BenchmarkRecord, OnlineDetectionTrace[Any]]]:
        """
        Execute the benchmark over all combinations of algorithms, data, and thresholds.

        Iterates through the combinations of algorithms, datasets, and thresholds.
        If disk caching (`dump_dir`) is enabled, it attempts to load previously
        calculated traces from the registry to bypass solver execution. If a trace
        is missing, it runs the solver, caches the resulting trace to disk, and
        updates the CSV registry.

        Returns
        -------
        list[tuple[BenchmarkRecord, OnlineDetectionTrace[Any]]]
            A list of execution results, where each element is a pair containing
            the benchmark metadata record and the corresponding detection trace.
        """
        results: list[tuple[BenchmarkRecord, OnlineDetectionTrace[Any]]] = []
        registry: dict[tuple[str, int, str, float], BenchmarkRecord] = {}
        registry_path: Path | None = None

        if self.__dump_dir is not None:
            self.__dump_dir.mkdir(parents=True, exist_ok=True)
            registry_path = self.__dump_dir / "benchmark_registry.csv"

            if registry_path.exists():
                with open(registry_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        record = BenchmarkRecord(
                            algorithm=row["algorithm"],
                            configuration_hash=int(row["configuration_hash"]),
                            data=row["data"],
                            threshold=float(row["threshold"]),
                            trace_path=row["trace_path"] if row["trace_path"] else None,
                        )
                        registry[record.key] = record

        for (algorithm, thresholds), provider in itertools.product(self.__algorithms, self.__providers):
            algo_name = str(algorithm)
            config_hash = hash(algorithm.configuration)
            data_name = provider.name

            for threshold in thresholds:
                key = (algo_name, config_hash, data_name, float(threshold))

                if key in registry:
                    cached_path = registry[key].trace_path
                    if cached_path is not None:
                        trace_file = Path(cached_path)
                        if trace_file.exists():
                            with open(trace_file, "rb") as f:
                                trace = pickle.load(f)
                            results.append((registry[key], trace))
                            continue

                steps = list(self.__solver.run(algorithm, provider, threshold))
                trace = OnlineDetectionTrace.from_run(steps, algo_name, config_hash)

                record = BenchmarkRecord(algo_name, config_hash, data_name, threshold, None)

                if self.__dump_dir is not None:
                    safe_data_name = "".join(c if c.isalnum() else "_" for c in data_name)
                    thr_str = "inf" if math.isinf(record.threshold) else f"{threshold:.4f}".replace(".", "_")
                    filename = f"{algo_name}_{config_hash}_{safe_data_name}_{thr_str}.pkl"

                    trace_path = self.__dump_dir / filename
                    with open(trace_path, "wb") as f:
                        pickle.dump(trace, f)

                    record.trace_path = str(trace_path)
                    registry[key] = record

                results.append((record, trace))

            if registry_path is not None:
                fieldnames = ["algorithm", "configuration_hash", "data", "threshold", "trace_path"]
                with open(registry_path, mode="w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for rec in registry.values():
                        writer.writerow(
                            {
                                "algorithm": rec.algorithm,
                                "configuration_hash": rec.configuration_hash,
                                "data": rec.data,
                                "threshold": rec.threshold,
                                "trace_path": rec.trace_path or "",
                            }
                        )

        return results
