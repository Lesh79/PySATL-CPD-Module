import tempfile
from os import walk
from pathlib import Path

import pytest

from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver


class TestGenerator:
    @pytest.mark.parametrize(
        "config_path_str,generator,expected_change_points_list,expected_lengths",
        (
            (
                "tests/test_CPDShell/test_configs/test_config_1.yml",
                ScipyDatasetGenerator(),
                [[], [], [20]],
                [20, 100, 40],
            ),
        ),
    )
    def test_generate_datasets(self, config_path_str, generator, expected_change_points_list, expected_lengths) -> None:
        generated = generator.generate_datasets(Path(config_path_str))
        for i in range(len(expected_lengths)):
            data_length = sum(1 for _ in generated[i][0])
            assert data_length == expected_lengths[i]
            assert generated[i][1] == expected_change_points_list[i]

    @pytest.mark.parametrize(
        "config_path_str,generator,expected_change_points_list,expected_lengths",
        (
            (
                "tests/test_CPDShell/test_configs/test_config_1.yml",
                ScipyDatasetGenerator(),
                [[], [], [20]],
                [20, 100, 40],
            ),
        ),
    )
    def test_generate_datasets_save(
        self, config_path_str, generator, expected_change_points_list, expected_lengths
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            saver = DatasetSaver(Path(tempdir), True)
            generated = generator.generate_datasets(Path(config_path_str), saver)
            for i in range(len(expected_lengths)):
                data_length = sum(1 for _ in generated[i][0])
                assert data_length == expected_lengths[i]
                assert generated[i][1] == expected_change_points_list[i]

            directory = [file_names for (_, _, file_names) in walk(tempdir)]
            for file_names in directory[1:]:
                assert sorted(file_names) == sorted(["changepoints.csv", "sample.adoc", "sample.png", "sample.csv"])
