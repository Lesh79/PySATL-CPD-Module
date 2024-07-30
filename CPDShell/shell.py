from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy

from CPDShell.Core.algorithms.graph_algorithm import Algorithm, GraphAlgorithm
from CPDShell.Core.cpd_core import CPDCore
from CPDShell.Core.scenario import Scenario
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.labeled_data import LabeledCPData


@dataclass
class CPContainer:
    """Container for results of CPD algorithms

    :param: result: list, containing change points, that were found by CPD algos
    :param: expected: list, containing expected change points, if it is needed
    """

    result: list[int]
    expected: list[int] | None


class CPDShell:
    """Class, that grants a convenient interface to
    work with CPD algorithms"""

    def __init__(
        self,
        data: Iterable[float | numpy.float64] | LabeledCPData,
        cpd_algorithm: Optional["Algorithm"] = None,
        scrubber_class: type[Scrubber] = Scrubber,
    ) -> None:
        """CPDShell object constructor

        :param: data: data for detection of CP
        :param: CPDalgorithm: CPD algorithm, that will search for change points
        :param: scrubber_class: class of preferable scrubber for splitting data into parts
        """
        self._data: Iterable[float | numpy.float64] | LabeledCPData = data
        scrubber_class = scrubber_class if scrubber_class is not None else Scrubber
        arg = 5
        cpd_algorithm = (
            cpd_algorithm if cpd_algorithm is not None else GraphAlgorithm(lambda a, b: abs(a - b) <= arg, 2)
        )
        self.cpd_core: CPDCore = CPDCore(
            scrubber_class(Scenario(10, True), data.raw_data if isinstance(data, LabeledCPData) else data),
            cpd_algorithm,
        )  # if no algo or scrubber was given, then some standard

    @property
    def data(self) -> Iterable[float | numpy.float64]:
        """Getter method for data param"""
        return self._data

    @data.setter
    def data(self, new_data: Sequence[float | numpy.float64]) -> None:
        """Setter method for changing data

        :param: new_data: new data, to replace the current one
        """
        self._data = new_data
        if isinstance(new_data, LabeledCPData):
            self.cpd_core.scrubber.data = new_data.raw_data
        else:
            self.cpd_core.scrubber.data = new_data

    @property
    def scrubber(self) -> Scrubber:
        """Getter method for scrubber"""
        return self.cpd_core.scrubber

    @scrubber.setter
    def scrubber(self, new_scrubber_class: type[Scrubber]) -> None:
        """Setter method for changing scrubber

        :param: new_scrubber_class: new scrubber, to replace the current one
        """
        self.cpd_core.scrubber = new_scrubber_class(self.cpd_core.scrubber.scenario, self._data)

    @property
    def CPDalgorithm(self) -> Algorithm:
        """Getter method for CPD algorithm param"""
        return self.cpd_core.algorithm

    @CPDalgorithm.setter
    def CPDalgorithm(self, new_algorithm: Algorithm) -> None:
        """Setter method for changing CPD algorithm

        :param: new_algorithm: new CPD algorithm, to replace the current one
        """
        self.cpd_core.algorithm = new_algorithm

    @property
    def scenario(self) -> Scenario:
        """Getter method for scenario param"""
        return self.cpd_core.scrubber.scenario

    @scenario.setter
    def scenario(self, new_scenario: Scenario) -> None:
        """Setter method for changing scenario

        :param: new_scenario: new scenario object, to replace the current one
        """
        self.cpd_core.scrubber.scenario = new_scenario

    def change_scenario(self, change_point_number: int, to_localize: bool = False) -> None:
        """Method for editing scenario

        :param: change_point_number: number of change points user wants to detect
        :param: to_localize: bool value that states if it is necessary to localize change points
        """
        self.cpd_core.scrubber.scenario = Scenario(change_point_number, to_localize)

    def run_cpd(self) -> CPContainer:
        """Execute CPD algorithm, returns its result and prints it

        :return: CPContainer object, containing algo result CP and expected CP if needed
        """
        algo_results = self.cpd_core.run()
        output = CPContainer(algo_results, None)
        if isinstance(self._data, LabeledCPData):
            output.expected = list(self._data.change_points)
        return output

    def print_cpd_results(self, exec_results: CPContainer) -> None:
        """prints results of run_CPD method in a pretty way

        :param: exec_results: output from run_cpd method, containing results and optional expected results
        """

        def _find_symm_diff(list1: list, list2: list) -> list:
            """helper function. Shows symm diff between two lists

            :param: list1: first list
            :param: list2: second list

            :return: list with symm diff of two lists
            """
            list1, list2 = set(list1), set(list2)
            return sorted(list(list1.symmetric_difference(list2)))

        result = exec_results.result
        expected = exec_results.expected
        if result is None:
            raise ValueError("wrong argument was given, result not found")
        result_output = ";".join(map(str, result))
        if expected is None:
            print(f"Located change points: ({result_output})")
            return
        expected_output = ";".join(map(str, expected))
        diff = ";".join(map(str, _find_symm_diff(result, expected)))
        print(f"Located change points: ({result_output})")
        print(f"Expected change point: ({expected_output})")
        print(f"Difference: ({diff})")
