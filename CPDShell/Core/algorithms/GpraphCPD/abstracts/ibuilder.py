from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph


class IBuilder(ABC):
    def __init__(self, data: Iterable[float], compare: Callable[[Any, Any], bool]):
        """
        Initialize the builder with data and a comparison function.

        :param data: List of elements to be used in building the graph.
        :param compare: Callable that takes two elements and returns a boolean indicating
                        if an edge should exist between them.
        """
        self.data = list(data)
        self.compare = compare
        self.num_of_edges: int = 0

    @abstractmethod
    def build_graph(self) -> IGraph:
        """
        Abstract method to build and return a graph representation.

        :return: An instance of IGraph representing the built graph.
        """
        pass
