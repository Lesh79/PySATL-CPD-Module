from typing import Any

from numpy import dtype, ndarray

from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph


class GraphMatrix(IGraph):
    def __init__(self, graph: ndarray[Any, dtype], num_of_edges: int):
        """
        Initialize the GraphMatrix with the adjacency matrix and number of edges.

        :param graph: Adjacency matrix representing the graph.
        :param num_of_edges: Number of edges in the graph.
        """
        super().__init__(num_of_edges, len(graph))
        self.mtx = graph

    def __getitem__(self, item):
        """
        Get the row of the adjacency matrix for a given node.

        :param item: Node index.
        :return: Row of the adjacency matrix corresponding to the node.
        """
        return self.mtx[item]

    def check_edges_exist(self, thao: int) -> int:
        count_edges = 0
        for node_before in range(thao):
            for node_after in range(thao, self.len):
                if self.mtx[node_before, node_after] == 1:
                    count_edges += 1
        return count_edges

    def sum_of_squares_of_degrees_of_nodes(self) -> int:
        sum_squares = 0
        for node_1 in range(0, self.len):
            node_degree = 0
            for node_2 in range(0, self.len):
                if self.mtx[node_1, node_2] == 1:
                    node_degree += 1
            node_degree = node_degree**2
            sum_squares += node_degree
        return sum_squares
