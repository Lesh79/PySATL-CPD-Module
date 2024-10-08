import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import numpy
from matplotlib import pyplot as plt

from CPDShell.Core.algorithms.graph_algorithm import Algorithm, GraphAlgorithm
from CPDShell.Core.cpd_core import CPDCore
from CPDShell.Core.scenario import Scenario
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.labeled_data import LabeledCPData


class CPContainer:
    """Container for results of CPD algorithms"""

    def __init__(
        self,
        data: Sequence[float | numpy.float64],
        result: list[int],
        expected_result: list[int] | None,
        time_sec: float,
    ) -> None:
        """Object constructor

        :param: result: list, containing change points, that were found by CPD algos
        :param: expected_result: list, containing expected change points, if it is needed
        :param: time_sec: a float number, time of CPD algo execution in fractional seconds
        """
        self.data = data
        self.result = result
        self.expected_result = expected_result
        self.time_sec = time_sec

    @property
    def result_diff(self) -> list:
        """method for calculation symmetrical diff between results and expected results (if its granted)

        :return: symmetrical difference between results and expected results
        """
        if self.expected_result is None:
            raise ValueError("this object is not provided with expected result, thus diff cannot be calculated.")
        first, second = set(self.result), set(self.expected_result)
        return sorted(list(first.symmetric_difference(second)))

    def __str__(self) -> str:
        """method for printing results of CPD algo results in a convenient way

        :return: string with brief CPD algo execution results
        """
        cp_results = ";".join(map(str, self.result))
        method_output = f"Located change points: ({cp_results})\n"
        if self.expected_result is not None:
            expected_cp_results = ";".join(map(str, self.expected_result))
            diff = ";".join(map(str, self.result_diff))
            method_output += f"Expected change point: ({expected_cp_results})\n"
            method_output += f"Difference: ({diff})\n"
        method_output += f"Computation time (sec): {round(self.time_sec, 2)}"
        return method_output

    def visualize(self, to_show: bool = True, output_directory: Path | None = None, name: str = "Graph") -> None:
        """method for building and analyzing graph

        :param to_show: is it necessary to show a graph
        :param output_directory: If necessary, the path to the directory to save the graph
        :param name: If necessary, graph name for saving
        """
        plt.plot(self.data)
        if self.expected_result is None:
            plt.vlines(x=self.result, ymin=min(self.data), ymax=max(self.data), colors="orange", ls="--")
            plt.gca().legend(("data", "detected"))
        else:
            correct, incorrect, undetected = set(), set(), set(self.expected_result)
            for point in self.result:
                if point in self.expected_result:
                    correct.add(point)
                    undetected.remove(point)
                elif point not in undetected:
                    incorrect.add(point)
            plt.vlines(x=list(correct), ymin=min(self.data), ymax=max(self.data), colors="green", ls="--")
            plt.vlines(x=list(incorrect), ymin=min(self.data), ymax=max(self.data), colors="red", ls="--")
            plt.vlines(x=list(undetected), ymin=min(self.data), ymax=max(self.data), colors="orange", ls="--")
            plt.gca().legend(("data", "correct detected", "incorrect detected", "undetected"))
        if output_directory:
            if not output_directory.exists():
                output_directory.mkdir()
            plt.savefig(output_directory.joinpath(Path(name)))
        if to_show:
            plt.show()


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
        """Execute CPD algorithm, returns ifrom dataclasses import dataclassts result and prints it

        :return: CPContainer object, containing algo result CP and expected CP if needed
        """
        time_start = time.perf_counter()
        algo_results = self.cpd_core.run()
        time_end = time.perf_counter()
        expected_change_points = self._data.change_points if isinstance(self._data, LabeledCPData) else None
        data = self._data.raw_data if isinstance(self._data, LabeledCPData) else self._data
        return CPContainer(data, algo_results, expected_change_points, time_end - time_start)
