from .abstract_algorithm import Algorithm


class GraphAlgorithm(Algorithm):
    def __init__(self, parameter_1: int, parameter_2: float):
        self.parameter_1 = parameter_1
        self.parameter_2 = parameter_2

    def localize(self, window: list[float]) -> list[int]:
        return [len(window) // 2]

    def detect(self, window: list[float]) -> list[int]:
        return [len(window), len(window)]
