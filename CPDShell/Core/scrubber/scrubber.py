from collections.abc import Sequence

from ..scenario import Scenario


class Scrubber:
    """A scrubber for dividing data into windows
    and subsequent processing of data windows
    by change point detection algorithms
    """

    def __init__(
            self,
            scenario: Scenario,
            data,
            window_length: int = 10,
            movement_k: float = 1 / 3,
    ) -> None:
        """A scrubber for dividing data into windows
        and subsequent processing of data windows
        by change point detection algorithms

        :param scenario: :class:`Scenario` object with information about the scrubber task
        :param data: list of values for change point detection
        :param window_length: length of data window
        :param movement_k: how far will the window move relative to the length
        """

        self.window_length = window_length
        self._movement_k = movement_k
        self.scenario = scenario
        self.data: Sequence[float] = data
        self.is_running = True
        self.change_points: list[int] = []
        self._next_window: tuple[int, int] | None = (0, self.window_length)

    def generate_window(self) -> Sequence[float]:
        """Function for dividing data into parts to feed into the change point detection algorithm

        :raises ValueError: all data has already been given
        :return: window (part of data) for change point detection algorithm
        """
        if not self.is_running or self._next_window is None:
            raise ValueError("All windows were given")
        window_start, window_end = self._next_window
        window = self.data[window_start:window_end]
        return window

    def add_change_points(self, window_change_points: list[int]) -> None:
        """Function for mapping window change points to global data

        :param window_change_points: change points in window
        :raises ValueError: all data windows have been processed
        """
        if self._next_window is None:
            raise ValueError("There are no windows to consider")
        for window_change_point in window_change_points:
            change_point = self._next_window[0] + window_change_point
            self.change_points.append(change_point)
            if len(self.change_points) == self.scenario.change_point_number:
                self._next_window = None
                self.is_running = False
                return

        if window_change_points:
            start, end = self.change_points[-1], self.change_points[-1] + self.window_length
            self._next_window = (start, end)
        else:
            delta = int(self._movement_k * self.window_length)
            start, end = self._next_window[0] + delta, self._next_window[1] + delta
            if end >= len(self.data):
                self._next_window = None
                self.is_running = False
            else:
                self._next_window = (start, end)

    def restart(self) -> None:
        self._next_window = (0, self.window_length)
        self.is_running = True
        self.change_points = []
