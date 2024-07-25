import os.path
from collections.abc import Iterable, Iterator
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

    def __init__(self, raw_data: Iterable[float], expected_res: Iterable[float]) -> None:
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


class CPDShell:
    """Class, that grants a convenient interface to
    work with CPD algorithms"""

    def __init__(
        self,
        data: Iterable[float],
        CPDalgorithm: "Algorithm" = GraphAlgorithm(1, 2),
        scrubber_class: type[Scrubber] = Scrubber,
    ) -> None:
        """CPDShell object constructor

        :param: data: data for detection of CP
        :param: CPDalgorithm: CPD algorithm, that will search for change points
        :param: scrubber_class: class of preferable scrubber for splitting data into parts
        """
        self._data: Iterable[float] | LabeledCPData = data
        self.cpd_core: CPDCore = CPDCore(
            scrubber_class(Scenario(9999999999), data), CPDalgorithm
        )  # if no algo or scrubber was given, then some standard

    @property
    def data(self) -> Iterable[float]:
        """Getter method for data param"""
        return self._data

    @data.setter
    def data(self, new_data: Iterable[float]) -> None:
        """Setter method for changing data

        :param: new_data: new data, to replace the current one
        """
        self._data = new_data
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

    def run_cpd(self) -> dict:  # (?) type of return and the way of printing result
        """Execute CPD algorithm, returns its result and prints it

        :return: dict with "result" field and optional "expected_results" field
        """
        algo_results = self.cpd_core.run()
        output = {"result": algo_results}
        if isinstance(self._data, LabeledCPData):
            output["expected"] = self._data.expected_res
        return output

    def print_cpd_results(self, exec_results: dict) -> None:
        """prints results of run_CPD method in a pretty way

        :param: exec_results: output from run_CPD method, dict, containing results and optional expected results
        """

        def _find_symm_diff(list1, list2) -> list:
            """helper function. Shows symm diff between two lists

            :param: list1: first list
            :param: list2: second list

            :return: list with symm diff of two lists
            """
            list1, list2 = set(list1), set(list2)
            return sorted(list(list1.symmetric_difference(list2)))

        result = exec_results.get("result")
        expected = exec_results.get("expected")
        if result is None:
            raise ValueError("wrong argument was given, result not found")
        result_output = ";".join(result)
        if expected is None:
            print(f"Located change points: ({result_output})")
            return
        expected_output = ";".join(expected)
        diff = ";".join(_find_symm_diff(result, expected))
        print(
            f"""Located change points: ({result_output})
Expected change point: ({expected_output})
Difference: ({diff})"""
        )
