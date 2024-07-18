from abc import ABC, abstractmethod
from collections.abc import Iterable


class Algorithm(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def detect(self, window: Iterable[float]) -> list[int]:
        # maybe rtype tuple[int]
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: list of right borders of window change points
        """
        ...

    @abstractmethod
    def localize(self, window: Iterable[float]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        ...
