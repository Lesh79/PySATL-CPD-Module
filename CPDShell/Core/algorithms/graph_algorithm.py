from typing import Iterable, Sized

from .abstract_algorithm import Algorithm


class GraphAlgorithm(Algorithm):
    def __init__(self, parameter_1: int, parameter_2: float):
        self.parameter_1 = parameter_1
        self.parameter_2 = parameter_2

    def localize(self, window: Iterable[float]) -> list[int]:
        return [0]

    def detect(self, window: Iterable[float]) -> list[int]:
        return [10]
