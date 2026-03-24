"""
Module contains online detection trace container for streaming changepoint detection.

This module provides containers for storing step-by-step results and aggregated
traces from online changepoint detection algorithms.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from pysatl_cpd._typing import Number, UnivariateNumericArray
from pysatl_cpd.detection_trace import DetectionTrace
from pysatl_cpd.online.ionline_algorithm import OnlineAlgorithmState


@dataclass(kw_only=True)
class OnlineDetectionStepResult[StateT: OnlineAlgorithmState]:
    """
    Result of processing a single observation in online changepoint detection.

    This class captures the complete output for one step of an online detection
    algorithm, including detection flags, computed statistics, and timing
    information.

    Parameters
    ----------
    step_num : int, default=0
        Zero-based index of the processed observation.
    is_change_point : bool, default=False
        Whether a changepoint was detected at this step.
    is_force_change_point : bool, default=False
        Whether a changepoint was forced due to maximum runlength constraint.
    is_in_skip_period : bool, default=False
        Whether this step occurred during a post-detection skip period.
    detection_function : Number, default=nan
        The value of the detection statistic computed for this observation.
    processing_time : Number, default=nan
        Wall-clock time in seconds spent processing this step.
    algorithm_state : OnlineAlgorithmState | None, default=None
        Snapshot of algorithm internal state after processing this step.
    """

    step_num: int = 0
    is_change_point: bool = False
    is_force_change_point: bool = False
    is_in_skip_period: bool = False
    detection_function: Number = float("nan")
    processing_time: Number = float("nan")
    algorithm_state: StateT | None = None


@dataclass(kw_only=True)
class OnlineDetectionTrace[StateT: OnlineAlgorithmState](DetectionTrace[Number]):
    """
    Complete trace of online changepoint detection execution.

    This class aggregates the results of running an online detection algorithm
    over a complete data sequence. It extends DetectionTrace with additional
    metadata specific to online detection, including per-step statistics,
    processing times, and forced detection markers.

    Parameters
    ----------
    threshold : Number | None, optional
        Detection threshold used during the run. Default is None.
    processing_time : UnivariateNumericArray
        Processing time for each observation step as a 1-D NumPy array.
    observation_scores : UnivariateNumericArray
        Detection function values for each observation as a 1-D NumPy array.
    algorithm_states : list[OnlineAlgorithmState | None]
        Algorithm state snapshots after processing each observation.
    detected_changes : list[int]
        Indices where changepoints were detected.
    skipped_observation : list[int], optional
        Indices where observations were skipped during post-detection periods.
        Default is empty list.
    forced_change_points : list[int], optional
        Indices where changepoints were forced due to maximum runlength.
        Default is empty list.
    """

    threshold: Number | None = None
    processing_time: UnivariateNumericArray
    observation_scores: UnivariateNumericArray
    algorithm_states: list[StateT | None]
    detected_changes: list[int]
    skipped_observation: list[int] = field(default_factory=list)
    forced_change_points: list[int] = field(default_factory=list)

    @classmethod
    def from_online_detection_steps(
        cls, threshold: Number | None, steps: Sequence[OnlineDetectionStepResult[StateT]]
    ) -> "OnlineDetectionTrace[StateT]":
        """
        Construct an OnlineDetectionTrace from a sequence of step results.

        This factory method aggregates per-step results into a complete trace,
        extracting detection indices, processing times, and state snapshots.

        Parameters
        ----------
        threshold : Number | None
            The detection threshold used during execution.
        steps : Sequence[OnlineDetectionStepResult]
            Sequence of step results from processing each observation.

        Returns
        -------
        OnlineDetectionTrace
            Aggregated trace containing all detection results and metadata.

        Examples
        --------
        >>> steps = [
        ...     OnlineDetectionStepResult(step_num=0, detection_function=0.1),
        ...     OnlineDetectionStepResult(step_num=1, detection_function=0.8, is_change_point=True)
        ... ]
        >>> trace = OnlineDetectionTrace.from_online_detection_steps(threshold=0.5, steps=steps)
        >>> trace.detected_changes
        [1]
        """
        step_nums = range(len(steps))

        # Extract detection function values into a float64 array
        detection_scores: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([step.detection_function for step in steps], dtype=np.float64)
        )

        # Extract processing times into a float64 array
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([step.processing_time for step in steps], dtype=np.float64)
        )

        # Extract algorithm states preserving None values
        algorithm_states = [step.algorithm_state for step in steps]

        # Identify indices of different detection types
        detected_indices = [idx for idx in step_nums if steps[idx].is_change_point]
        skipped_indices = [idx for idx in step_nums if steps[idx].is_in_skip_period]
        forced_indices = [idx for idx in step_nums if steps[idx].is_force_change_point]

        return cls(
            threshold=threshold,
            observation_scores=detection_scores,
            processing_time=processing_times,
            algorithm_states=algorithm_states,
            detected_changes=detected_indices,
            skipped_observation=skipped_indices,
            forced_change_points=forced_indices,
        )
