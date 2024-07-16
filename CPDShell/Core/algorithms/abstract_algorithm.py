from abc import ABC, abstractmethod


class Algorithm(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def detect(self, window: list[float]) -> list[int]:
        # maybe rtype tuple[int]
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :type window: list[float]
        :return: list of right borders of window change points
        :rtype: list[int]
        """
        ...

    @abstractmethod
    def localize(self, window: list[float]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :type window: list[float]
        :return: list of window change points
        :rtype: list[int]
        """
        ...
