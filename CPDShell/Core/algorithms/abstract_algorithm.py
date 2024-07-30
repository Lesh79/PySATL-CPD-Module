from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy


class Algorithm(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def detect(self, window: Iterable[float | numpy.float64]) -> int:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        ...

    @abstractmethod
    def localize(self, window: Iterable[float | numpy.float64]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        ...
