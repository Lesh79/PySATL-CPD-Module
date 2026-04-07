import csv
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
    algorithm: str
    configuration_hash: str
    data: str
    threshold: float
    trace_path: str | None = None

    @property
    def key(self) -> tuple[str, str, str, float]:
        return (self.algorithm, self.configuration_hash, self.data, self.threshold)


class BenchmarkExecutor[DataT]:
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
        results: list[tuple[BenchmarkRecord, OnlineDetectionTrace[Any]]] = []
        registry: dict[tuple[str, str, str, float], BenchmarkRecord] = {}
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
                            configuration_hash=row["configuration_hash"],
                            data=row["data"],
                            threshold=float(row["threshold"]),
                            trace_path=row["trace_path"] if row["trace_path"] else None,
                        )
                        registry[record.key] = record

        for algorithm, thresholds in self.__algorithms:
            algo_name = str(algorithm)
            config_hash = str(hash(algo_name))

            for provider in self.__providers:
                data_name = provider.name

                for threshold in thresholds:
                    key = (algo_name, config_hash, data_name, float(threshold))

                    if key in registry and registry[key].trace_path:
                        trace_file = Path(registry[key].trace_path)  # type: ignore
                        if trace_file.exists():
                            with open(trace_file, "rb") as f:
                                trace = pickle.load(f)
                            results.append((registry[key], trace))
                            continue

                    steps = list(self.__solver.run(algorithm, provider, threshold))
                    trace = OnlineDetectionTrace.from_run(steps)

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
