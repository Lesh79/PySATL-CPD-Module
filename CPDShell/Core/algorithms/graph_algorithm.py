from collections.abc import Callable, Iterable
from typing import Any

from .abstract_algorithm import Algorithm
from .GpraphCPD.Builders.matrix_builder import AdjacencyMatrixBuilder
from .GpraphCPD.graph_cpd import GraphCPD


class GraphAlgorithm(Algorithm):
    def __init__(self, compare_func: Callable[[Any, Any], bool], threshold: float):
        self.compare = compare_func
        self.threshold = threshold

    def localize(self, window: Iterable[float]) -> list[int]:
        builder = AdjacencyMatrixBuilder(window, self.compare)
        graph = builder.build_graph()
        cpd = GraphCPD(graph)
        num_cpd: list[int] = cpd.find_changepoint(self.threshold)
        return num_cpd

    def detect(self, window: Iterable[float]) -> int:
        builder = AdjacencyMatrixBuilder(window, self.compare)
        graph = builder.build_graph()
        cpd = GraphCPD(graph)
        num_cpd: list[int] = cpd.find_changepoint(self.threshold)
        return len(num_cpd)
