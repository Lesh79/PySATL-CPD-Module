from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class NoResetDetectionTrace[StateT: OnlineAlgorithmState](OnlineDetectionTrace[StateT]):
    @classmethod
    def from_inf_trace(
        cls,
        source_trace: OnlineDetectionTrace[StateT],
        detected_change_points: list[int],
        threshold: float,
    ) -> "NoResetDetectionTrace[StateT]":
        raise NotImplementedError("Method 'from_inf_trace' is not implemented yet.")
