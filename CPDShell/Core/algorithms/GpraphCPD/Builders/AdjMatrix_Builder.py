from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from CPDShell.Core.algorithms.GpraphCPD.abstracts.ibuilder import IBuilder
from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph
from CPDShell.Core.algorithms.GpraphCPD.GraphMatrix import GraphMatrix


class AdjacencyMatrixBuilder(IBuilder):
    def __init__(self, data: Iterable[float], comparing_function: Callable[[Any, Any], bool]):
        super().__init__(data, comparing_function)

    def build_matrix(self) -> np.ndarray:  # Adjacency Matrix
        """
        Build the adjacency matrix from the provided data.

        :return: A NumPy ndarray representing the adjacency matrix where element [i, j] is 1 if
                 there is an edge between node i and node j, otherwise 0.
        """
        count_edges = 0
        count_nodes = len(self.data)
        adjacency_matrix = np.zeros((count_nodes, count_nodes), dtype=int)

        for i in range(count_nodes):
            for j in range(count_nodes):
                if self.compare(self.data[i], self.data[j]) and (i != j):
                    adjacency_matrix[i, j] = 1
                    count_edges += 1
        self.num_of_edges = count_edges // 2

        return adjacency_matrix

    def build_graph(self) -> IGraph:
        graph = self.build_matrix()
        return GraphMatrix(graph, self.num_of_edges)
