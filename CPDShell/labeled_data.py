import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy

from CPDShell.generator.generator import DatasetGenerator, ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver


class LabeledCPData:
    """Class for generating and storing labeled data,
    needed in CPDShell"""

    def __init__(self, raw_data: Sequence[float | numpy.float64], change_points: Sequence[int]) -> None:
        """LabeledCPData object constructor

        :param: raw_data: data, that will be passed into CPD algo
        :param: change_points: expected results after passing raw_data into CPD algo
        """
        self.raw_data = raw_data
        self.change_points = change_points

    def __iter__(self) -> Iterator:
        """labeledCPData iterator"""
        return self.raw_data.__iter__()

    def __str__(self) -> str:
        """Shows main info about LabeledCPData object"""
        return f"data={self.raw_data}, change_points={self.change_points}"

    @staticmethod
    def generate_cp_datasets(
        config_path: Path,
        generator: DatasetGenerator = ScipyDatasetGenerator(),
        to_save: bool = False,
        output_directory: Path = Path(),
        to_replace: bool = True,
    ) -> dict[str, "LabeledCPData"]:
        """Method for generating labeled data, that contains CP with specific
        distribution

        :param config_path: path to config file
        :param generator: DataGenerator object, defaults to ScipyDatasetGenerator()
        :param to_save: is it necessary to save the data, defaults to False
        :param output_directory: directory to save data, defaults to Path()
        :param to_replace: is it necessary to replace the files in directory

        :return: dict of pairs: name, LabeledCPData (pairs of data and change points)"""
        # maybe create default config
        if not os.path.exists(config_path):
            raise ValueError("Incorrect config path")
        if to_save:
            datasets = generator.generate_datasets(config_path, DatasetSaver(output_directory, to_replace))
        else:
            datasets = generator.generate_datasets(config_path)
        labeled_data_dict = dict()
        for name in datasets:
            data, change_points = datasets[name]
            labeled_data_dict[name] = LabeledCPData(data, change_points)
        return labeled_data_dict

    @staticmethod
    def read_generated_datasets(datasets_directory: Path) -> dict[str, "LabeledCPData"]:
        """Read already generated datasets from directory

        :param datasets_directory: directory with datasets
        :return: dict of pairs: name, LabeledCPData (pairs of data and change points)"""
        datasets = dict()
        for dataset_directory in os.scandir(datasets_directory):
            dataset_files = dict()
            with os.scandir(dataset_directory) as entries:
                for file in entries:
                    dataset_files[file.name] = file
            if "changepoints.csv" not in dataset_files or "sample.csv" not in dataset_files:
                raise ValueError(f"{datasets_directory} is not datasets directory")
            with open(dataset_files["sample.csv"]) as sample:
                data = list(map(numpy.float64, sample.readlines()))
            with open(dataset_files["changepoints.csv"]) as changepoints:
                change_points = list(map(int, changepoints.readlines()))
            datasets[dataset_directory.name] = LabeledCPData(data, change_points)
        return datasets
