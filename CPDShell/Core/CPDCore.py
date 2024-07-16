from .algorithms.graph_algorithm import *
from .scrubber.scrubber import *


class CPDCore:
    """Change Point Detection Core"""

    def __init__(self, scrubber: Scrubber, algorithm: Algorithm) -> None:
        """Change Point Detection Core

        :param scrubber: scrubber for dividing data into windows
            and subsequent processing of data windows
            by change point detection algorithms
        :type scrubber: :class:'Scrubber'
        :param algorithm: change point detection algorithm
        :type algorithm: :class:'Algorithm'
        """
        self.scrubber = scrubber
        self.algorithm = algorithm

    def run(self) -> list[int]:
        """Find change points

        :return: list of change points
        :rtype: list[int]
        """
        while self.scrubber.is_running:
            window = self.scrubber.generate_window()
            if self.scrubber.scenario.to_localize:
                window_change_points = self.algorithm.localize(window)
            else:
                window_change_points = self.algorithm.detect(window)
            self.scrubber.add_change_points(window_change_points)
        return self.scrubber.change_points
