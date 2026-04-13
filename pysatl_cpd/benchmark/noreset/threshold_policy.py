from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pysatl_cpd.core.typedefs import UnivariateNumericArray


@runtime_checkable
class ThresholdPolicy(Protocol):
    def apply(
        self,
        detection_function: UnivariateNumericArray,
        threshold: float,
        change_points: Sequence[int],  # true, 1-based
    ) -> list[int]: ...  # 1-based signal indices


class PointBasedPolicy:
    def __init__(self, strict: bool = True) -> None:
        return

    def apply(
        self,
        detection_function: UnivariateNumericArray,
        threshold: float,
        change_points: Sequence[int],  # true, 1-based
    ) -> list[int]:
        raise NotImplementedError("Method `apply` is not implemented yet.")


class EventBasedPolicy:
    def __init__(
        self,
        max_delay: int,
        strict_edge: bool = True,
        strict_point: bool = True,
    ) -> None:
        return

    def apply(
        self,
        detection_function: UnivariateNumericArray,
        threshold: float,
        change_points: Sequence[int],  # true, 1-based
    ) -> list[int]:
        raise NotImplementedError("Method `apply` is not implemented yet.")
