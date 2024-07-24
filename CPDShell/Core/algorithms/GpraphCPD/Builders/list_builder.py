from collections.abc import Callable
from typing import Any

from CPDShell.Core.algorithms.GpraphCPD.abstracts.ibuilder import IBuilder
from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph
from CPDShell.Core.algorithms.GpraphCPD.graph_list import GraphList


class AdjacencyListBuilder(IBuilder):
    def __init__(self, data: list[Any], comparing_function: Callable[[Any, Any], bool]):
        super().__init__(data, comparing_function)

    def build(self) -> dict[int, list]:  # Adjacency List
        """
        Build the adjacency list from the provided data.

        :return: A dictionary representing the adjacency list where keys are node indices and values
                 are lists of adjacent nodes.
        """
        unique_edges = set()
        count_nodes = len(self.data)
        adjacency_list: dict[int, list] = {index: [] for index in range(count_nodes)}
        for i in range(count_nodes):
            for j in range(count_nodes):
                if self.compare(self.data[i], self.data[j]) and (i != j):
                    adjacency_list[i].append(self.data[j])
                    edge = tuple(sorted((i, j)))
                    unique_edges.add(edge)
        self.num_of_edges = len(unique_edges)

        # for i in range(0, len(self.data)):
        #     print(f"{self.data[i]}: {adjacency_list[i]}")

        return adjacency_list

    def build_graph(self) -> IGraph:
        graph = self.build()
        return GraphList(graph, self.data, self.num_of_edges)
