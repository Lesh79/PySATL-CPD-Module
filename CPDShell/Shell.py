from typing import Optional, Iterable, Iterator

from CPDShell.Core.CPDCore import *
from CPDShell.Core.scrubber.scrubber import Scrubber


class MarkedCPData:
    """Class for generating and storing marked data,
    needed in CPDShell"""

    def __init__(self, raw_data: Iterable[float], expected_res) -> None:  # (?) type of expected_res
        """MarkedCPData object constructor"""
        self.raw_data = raw_data
        self.expected_res = expected_res

    def __iter__(self) -> Iterator:
        """MarkedCPData iterator"""
        return self.raw_data.__iter__()

    def generate_CP_dataset(self, distribution) -> "MarkedCPData":  # (?) type of distribution
        """Method for generating marked data, that contains CP with specific
        distribution"""
        # (?) distribution mb optional.
        return MarkedCPData([], [])


class CPDShell:
    """Class, that grants a convenient interface to
    work with CPD algorithms"""

    def __init__(
        self,
        data: Iterable,
        *,
        algorithm: Optional["Algorithm"] = None,
        scrubber_class: Optional[type[Scrubber]] = None
    ) -> None:
        """CPDShell object constructor"""
        self._data = data
        scrubber_class = scrubber_class if scrubber_class is not None else Scrubber
        algorithm = algorithm if algorithm is not None else GraphAlgorithm(1, 2)
        self.cpd_core: CPDCore = CPDCore(
            scrubber_class(Scenario(10, True), data), algorithm
        )  # if no algo or scrubber was given, then some standard

    @property
    def data(self) -> Iterable[float]:
        return self._data

    @data.setter
    def data(self, new_data: Iterable[float]) -> None:
        self._data = new_data
        self.cpd_core.scrubber.data = new_data

    @property
    def scrubber(self) -> Scrubber:
        return self.cpd_core.scrubber

    @scrubber.setter
    def scrubber(self, new_scrubber: type[Scrubber]) -> None:
        self.cpd_core.scrubber = new_scrubber(self.cpd_core.scrubber.scenario, self._data)

    @property
    def CPDalgorithm(self) -> Algorithm:
        return self.cpd_core.algorithm

    @CPDalgorithm.setter
    def CPDalgorithm(self, new_algorithm: type[Algorithm]) -> None:
        self.cpd_core.algorithm = new_algorithm()

    @property
    def scenario(self) -> Scenario:
        return self.cpd_core.scrubber.scenario

    @scenario.setter
    def scenario(self, new_scenario: Scenario) -> None:
        self.cpd_core.scrubber.scenario = new_scenario

    def run_CPD(self) -> dict:  # (?) type of return and the way of printing result
        """Execute CPD algorithm, returns its result and prints it"""
        algo_result = self.cpd_core.run()  # TODO: rename later
        result = {"result": algo_result}
        if isinstance(self._data, MarkedCPData):
            result["expected"] = self._data.expected_res
        return result
