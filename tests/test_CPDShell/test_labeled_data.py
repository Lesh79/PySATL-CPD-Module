import tempfile
from os import walk
from pathlib import Path

import pytest

from CPDShell.labeled_data import LabeledCPData


class TestLabeledCPData:
    data = LabeledCPData([1, 2, 3], [4, 5, 6])

    def test_init(self) -> None:
        assert self.data.raw_data == [1, 2, 3]
        assert self.data.change_points == [4, 5, 6]

    def test_iter(self) -> None:
        assert list(self.data.__iter__()) == [1, 2, 3]

    @pytest.mark.parametrize(
        "config_path_str,expected_change_points_list,expected_lengths",
        (
            (
                "tests/test_CPDShell/test_configs/test_config_1.yml",
                {
                    "20-normal-0-1-20-normal-10-1": [20],
                    "20-normal-0-1-no-change-point": [],
                    "100-normal-0-1-no-change-point": [],
                },
                {
                    "20-normal-0-1-20-normal-10-1": 40,
                    "20-normal-0-1-no-change-point": 20,
                    "100-normal-0-1-no-change-point": 100,
                },
            ),
        ),
    )
    def test_generate_datasets(self, config_path_str, expected_change_points_list, expected_lengths) -> None:
        generated = LabeledCPData.generate_cp_datasets(Path(config_path_str))
        for name in expected_lengths:
            data_length = len(generated[name].raw_data)
            assert data_length == expected_lengths[name]
            assert generated[name].change_points == expected_change_points_list[name]

    @pytest.mark.parametrize(
        "config_path_str,expected_change_points_list,expected_lengths",
        (
            (
                "tests/test_CPDShell/test_configs/test_config_1.yml",
                {
                    "20-normal-0-1-20-normal-10-1": [20],
                    "20-normal-0-1-no-change-point": [],
                    "100-normal-0-1-no-change-point": [],
                },
                {
                    "20-normal-0-1-20-normal-10-1": 40,
                    "20-normal-0-1-no-change-point": 20,
                    "100-normal-0-1-no-change-point": 100,
                },
            ),
        ),
    )
    def test_generate_datasets_save(self, config_path_str, expected_change_points_list, expected_lengths) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            generated = LabeledCPData.generate_cp_datasets(
                Path(config_path_str), to_save=True, output_directory=Path(tempdir)
            )
            for name in expected_lengths:
                data_length = len(generated[name].raw_data)
                assert data_length == expected_lengths[name]
                assert generated[name].change_points == expected_change_points_list[name]

            directory = [file_names for (_, _, file_names) in walk(tempdir)]
            for file_names in directory[1:]:
                assert sorted(file_names) == sorted(["changepoints.csv", "sample.adoc", "sample.png", "sample.csv"])

    @pytest.mark.parametrize(
        "config_path_str",
        ("tests/test_CPDShell/test_configs/test_config_1.yml",),
    )
    def test_read_generated_datasets(self, config_path_str):
        with tempfile.TemporaryDirectory() as tempdir:
            generated = LabeledCPData.generate_cp_datasets(
                Path(config_path_str), to_save=True, output_directory=Path(tempdir)
            )
            read = LabeledCPData.read_generated_datasets(Path(tempdir))
            for name in generated:
                assert read[name].raw_data == generated[name].raw_data
                assert read[name].change_points == generated[name].change_points
