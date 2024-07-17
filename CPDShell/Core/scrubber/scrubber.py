from typing import Iterable, Optional

from ..scenario import Scenario


class Scrubber:
    """A scrubber for dividing data into windows
    and subsequent processing of data windows
    by change point detection algorithms
    """

    def __init__(
        self,
        scenario: Scenario,
        data: Iterable[float],
    ) -> None:
        """A scrubber for dividing data into windows
        and subsequent processing of data windows
        by change point detection algorithms

        :param scenario: :class:`Scenario` object with information about the scrubber task
        :param data: list of values for change point detection
        """

        # data: list or numpy.array
        self.scenario = scenario
        self._data = data
        # mock realization
        self.change_points: list[int] = []
        self.is_running = True
        self._uncompleted_window_index: list[int] = [0, 100, 200]
        self._completed_window_index: list[int] = [0]

    def generate_window(self) -> Iterable[float]:
        """Function for dividing data into parts to feed into the change point detection algorithm

        :raises ValueError: all data has already been given
        :return: window (part of data) for change point detection algorithm
        """
        if not self.is_running:
            raise ValueError("All windows were given")
        first_data_index = self._uncompleted_window_index[len(self._completed_window_index) - 1]
        second_data_index = self._uncompleted_window_index[len(self._completed_window_index)]
        window = self._data
        return window

    def add_change_points(self, window_change_points: list[int]) -> None:
        """Function for mapping window change points to global data

        :param window_change_points: change points in window
        """
        for window_change_point in window_change_points:
            if len(self.change_points) < self.scenario.change_point_number:
                self.change_points.append(self._completed_window_index[-1] + window_change_point)
            else:
                self._completed_window_index = self._uncompleted_window_index
                self.is_running = False
                return
        self._completed_window_index.append(self._uncompleted_window_index[len(self._completed_window_index)])
        if self._completed_window_index == self._uncompleted_window_index:
            self.is_running = False
