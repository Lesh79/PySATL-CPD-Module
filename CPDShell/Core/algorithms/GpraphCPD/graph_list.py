from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph


class GraphList(IGraph):
    def __init__(self, graph, data: list[float], num_of_edges: int):
        """
        Initialize the GraphList with the adjacency list, data, and number of edges.

        :param graph: Adjacency list representing the graph.
        :param data: List of elements representing the nodes.
        :param num_of_edges: Number of edges in the graph.
        """
        super().__init__(num_of_edges, len(data))
        self.graph = graph
        self.data = data

    def __getitem__(self, item):
        """
        Get the list of adjacent nodes for a given node.

        :param item: Node index.
        :return: List of adjacent nodes.
        """
        return self.graph[item]

    def check_edges_exist(self, thao: int) -> int:
        count_edges = 0
        for node_1 in range(thao):
            for node_2 in range(thao, self.len):
                if self.data[node_2] in self.graph[node_1]:
                    count_edges += 1
        return count_edges

    def sum_of_squares_of_degrees_of_nodes(self) -> int:
        sum_squares = 0
        for node in range(0, self.len):
            sum_squares += len(self.graph[node]) ** 2
        return sum_squares
