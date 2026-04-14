# -*- coding: ascii -*-
"""
Logging utilities for benchmark execution.
"""

import logging
from typing import Any

__author__ = "PySATL contributors"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class BenchmarkLogger:
    """Dedicated logger for benchmark execution with structured logging."""

    def __init__(self, name: str = "pysatl.benchmark"):
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup logger if not already configured."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message with optional context."""
        if kwargs:
            msg = f"{msg} | {' | '.join(f'{k}={v}' for k, v in kwargs.items())}"
        self.logger.info(msg)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with optional context."""
        if kwargs:
            msg = f"{msg} | {' | '.join(f'{k}={v}' for k, v in kwargs.items())}"
        self.logger.debug(msg)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        if kwargs:
            msg = f"{msg} | {' | '.join(f'{k}={v}' for k, v in kwargs.items())}"
        self.logger.warning(msg)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message."""
        if kwargs:
            msg = f"{msg} | {' | '.join(f'{k}={v}' for k, v in kwargs.items())}"
        self.logger.error(msg)

    def start_benchmark(
        self,
        n_algorithms: int,
        n_providers: int,
        n_total_runs: int,
    ) -> None:
        """Log benchmark start."""
        self.info(
            "Starting benchmark execution",
            algorithms=n_algorithms,
            providers=n_providers,
            total_runs=n_total_runs,
        )

    def algorithm_start(self, algo_name: str, n_thresholds: int) -> None:
        """Log algorithm processing start."""
        self.info(
            f"Processing algorithm: {algo_name}",
            thresholds=n_thresholds,
        )

    def threshold_processed(
        self,
        algo_name: str,
        threshold: float,
        n_providers: int,
    ) -> None:
        """Log threshold processing."""
        self.debug(
            "Threshold processed",
            algo=algo_name,
            threshold=f"{threshold:.4f}",
            providers=n_providers,
        )

    def cache_hit(self, algo_name: str, threshold: float, provider: str) -> None:
        """Log cache hit."""
        self.debug(
            "Cache hit",
            algo=algo_name,
            threshold=f"{threshold:.4f}",
            provider=provider,
        )

    def solver_start(self, algo_name: str, provider: str, threshold: float) -> None:
        """Log solver execution start."""
        self.debug(
            "Executing solver",
            algo=algo_name,
            provider=provider,
            threshold=f"{threshold:.4f}",
        )

    def metrics_computed(
        self,
        algo_name: str,
        threshold: float,
        metric_names: list[str],
    ) -> None:
        """Log metrics computation."""
        self.debug(
            "Metrics computed",
            algo=algo_name,
            threshold=f"{threshold:.4f}",
            metrics=", ".join(metric_names),
        )

    def benchmark_complete(self, total_runs: int, elapsed_sec: float) -> None:
        """Log benchmark completion."""
        avg_time = elapsed_sec / total_runs if total_runs > 0 else 0
        self.info(
            "Benchmark completed",
            total_runs=total_runs,
            elapsed_time=f"{elapsed_sec:.2f}s",
            avg_time_per_run=f"{avg_time:.3f}s",
        )

    def warning_no_metrics(self) -> None:
        """Log warning about missing metrics."""
        self.warning("No metrics registered for evaluation")

    def error_exception(self, algo_name: str, threshold: float, error: str) -> None:
        """Log exception during benchmark."""
        self.error(
            "Error during execution",
            algo=algo_name,
            threshold=f"{threshold:.4f}",
            error=error,
        )
