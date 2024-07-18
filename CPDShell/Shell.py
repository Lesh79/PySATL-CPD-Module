import os.path
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional

from CPDShell.Core.algorithms.graph_algorithm import Algorithm, GraphAlgorithm
from CPDShell.Core.CPDCore import CPDCore
from CPDShell.Core.scenario import Scenario
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.generator.generator import DatasetGenerator, ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver


class LabeledCPData:
    """Class for generating and storing labeled data,
    needed in CPDShell"""

    def __init__(self, raw_data: Iterable[float], expected_res) -> None:  # (?) type of expected_res
        """labeledCPData object constructor"""
        self.raw_data = raw_data
        self.expected_res = expected_res

    def __iter__(self) -> Iterator:
        """labeledCPData iterator"""
        return self.raw_data.__iter__()

    def __str__(self) -> str:
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


class CPDShell:
    """Class, that grants a convenient interface to
    work with CPD algorithms"""

    def __init__(
        self,
        data: Iterable[float],
        *,
        algorithm: Optional["Algorithm"] = None,
        scrubber_class: type[Scrubber] | None = None,
    ) -> None:
        """CPDShell object constructor"""
        self._data: Iterable[float] | LabeledCPData = data
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
        if isinstance(self._data, LabeledCPData):
            result["expected"] = self._data.expected_res
        return result


shell_labeled_data = CPDShell(LabeledCPData([1, 2, 3], [4, 5, 6]))
