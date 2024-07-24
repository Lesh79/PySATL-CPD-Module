from abc import ABC, abstractmethod

from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph


class IGraphCPD(ABC):
    def __init__(self, graph: IGraph):
        """
        Initialize the IGraphCPD with the given graph.

        :param graph: An instance of IGraph representing the graph.
        """
        self.graph = graph
        self.size = graph.len

    @abstractmethod
    def calculation_e(self, thao: int) -> float:
        """
        Calculate the mathematical expectation (E) using the given formula.

        :param thao: Index dividing the nodes into two sets.
        :return: Calculated expectation value.
        """
        pass

    @abstractmethod
    def calculation_var(self, thao: int) -> float:
        """
        Calculate the variance using the given formula.

        :param thao: Index dividing the nodes into two sets.
        :return: Calculated variance value.
        """
        pass

    @abstractmethod
    def calculation_z(self, thao: int) -> float:
        """
        Calculate the Z statistic.

        :param thao: Index dividing the nodes into two sets.
        :return: Calculated Z statistic.
        """
        pass

    @abstractmethod
    def find_changepoint(self, border: float) -> list:
        """
        Find change points in the data based on the Z statistic.

        :param border: Threshold value for detecting change points.
        :return: List of detected change points.
        """
        pass
