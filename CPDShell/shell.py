import os.path
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from CPDShell.Core.algorithms.graph_algorithm import Algorithm, GraphAlgorithm
from CPDShell.Core.cpd_core import CPDCore
from CPDShell.Core.scenario import Scenario
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.generator.generator import DatasetGenerator, ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver


class LabeledCPData:
    """Class for generating and storing labeled data,
    needed in CPDShell"""

    def __init__(self, raw_data: Sequence[float], expected_res) -> None:  # (?) type of expected_res
        """LabeledCPData object constructor

        :param: raw_data: data, that will be passed into CPD algo
        :param: expected_res: expected results after passing raw_data into CPD algo
        """
        self.raw_data = raw_data
        self.expected_res = expected_res

    def __iter__(self) -> Iterator:
        """labeledCPData iterator"""
        return self.raw_data.__iter__()

    def __str__(self) -> str:
        """Shows main info about LabeledCPData object"""
        return f"data={self.raw_data}, change_points={self.expected_res}"

    @staticmethod
    def generate_cp_dataset(
        config_path: Path,
        generator: DatasetGenerator = ScipyDatasetGenerator(),
        to_save: bool = False,
        output_directory: Path = Path(),
        to_replace: bool = True,
    ) -> list["LabeledCPData"]:
        """Method for generating labeled data, that contains CP with specific
        distribution

        :param config_path: path to config file
        :param generator: DataGenerator object, defaults to ScipyDatasetGenerator()
        :param to_save: is it necessary to save the data, defaults to False
        :param output_directory: directory to save data, defaults to Path()
        :param to_replace: is it necessary to replace the files in directory

        :return: list of LabeledCPData (pairs of data and change points)"""
        # maybe create default config
        if not os.path.exists(config_path):
            raise ValueError("Incorrect config path")
        if to_save:
            datasets = generator.generate_datasets(config_path, DatasetSaver(output_directory, to_replace))
        else:
            datasets = generator.generate_datasets(config_path)
        labeled_data_list = []
        for data, change_points in datasets:
            labeled_data_list.append(LabeledCPData(data, change_points))
        return labeled_data_list


class CPContainer:
    """Container for results of CPD algorithms"""

    def __init__(self, result: list[float], expected_result: list[float] | None, time_ms: float) -> None:
        """Object constructor

        :param: result: list, containing change points, that were found by CPD algos
        :param: expected_result: list, containing expected change points, if it is needed
        :param: time_ms: a float number, time of CPD algo execution in milliseconds
        """
        self.result = result
        self.expected_result = expected_result
        self.time_ms = time_ms

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
        method_output += f"Computation time (ms): {round(self.time_ms, 2)}"
        return method_output


class CPDShell:
    """Class, that grants a convenient interface to
    work with CPD algorithms"""

    def __init__(
        self,
        data: Iterable[float],
        cpd_algorithm: "Algorithm" = GraphAlgorithm(1, 2),
        scrubber_class: type[Scrubber] = Scrubber,
    ) -> None:
        """CPDShell object constructor

        :param: data: data for detection of CP
        :param: CPDalgorithm: CPD algorithm, that will search for change points
        :param: scrubber_class: class of preferable scrubber for splitting data into parts
        """
        self._data: Iterable[float] | LabeledCPData = data
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
    def data(self) -> Iterable[float]:
        """Getter method for data param"""
        return self._data

    @data.setter
    def data(self, new_data: Sequence[float]) -> None:
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
        expected_res = self._data.expected_res if isinstance(self._data, LabeledCPData) else None
        return CPContainer(algo_results, expected_res, time_end - time_start)
