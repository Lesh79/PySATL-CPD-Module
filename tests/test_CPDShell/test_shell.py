import tempfile
from os import walk
from pathlib import Path

import pytest

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.shell import CPDShell, LabeledCPData


class TestMarkedCPData:
    data = LabeledCPData([1, 2, 3], [4, 5, 6])

    def test_init(self) -> None:
        assert self.data.raw_data == [1, 2, 3]
        assert self.data.expected_res == [4, 5, 6]

    def test_iter(self) -> None:
        assert list(self.data.__iter__()) == [1, 2, 3]

    @pytest.mark.parametrize(
        "config_path_str,expected_change_points_list,expected_lengths",
        (("tests/test_CPDShell/test_configs/test_config_1.yml", [[], [], [20]], [20, 100, 40]),),
    )
    def test_generate_datasets(self, config_path_str, expected_change_points_list, expected_lengths) -> None:
        generated = LabeledCPData.generate_cp_dataset(Path(config_path_str))
        for i in range(len(expected_lengths)):
            data_length = sum(1 for _ in generated[i].raw_data)
            assert data_length == expected_lengths[i]
            assert generated[i].expected_res == expected_change_points_list[i]

    @pytest.mark.parametrize(
        "config_path_str,expected_change_points_list,expected_lengths",
        (("tests/test_CPDShell/test_configs/test_config_1.yml", [[], [], [20]], [20, 100, 40]),),
    )
    def test_generate_datasets_save(self, config_path_str, expected_change_points_list, expected_lengths) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            generated = LabeledCPData.generate_cp_dataset(
                Path(config_path_str), to_save=True, output_directory=Path(tempdir)
            )
            for i in range(len(expected_lengths)):
                data_length = sum(1 for _ in generated[i].raw_data)
                assert data_length == expected_lengths[i]
                assert generated[i].expected_res == expected_change_points_list[i]

            directory = [file_names for (_, _, file_names) in walk(tempdir)]
            for file_names in directory[1:]:
                assert sorted(file_names) == sorted(["changepoints.csv", "sample.adoc", "sample.png", "sample.csv"])


def custom_comparison(node1, node2):  # TODO: Remove it everywhere
    arg = 1
    return abs(node1 - node2) <= arg


class TestCPDShell:
    shell_normal = CPDShell([1, 2, 3, 4], cpd_algorithm=GraphAlgorithm(custom_comparison, 4), scrubber_class=Scrubber)
    shell_default = CPDShell([3, 4, 5, 6], cpd_algorithm=GraphAlgorithm(custom_comparison, 4))
    shell_marked_data = CPDShell(
        LabeledCPData([1, 2, 3, 4], [4, 5, 6, 7]), cpd_algorithm=GraphAlgorithm(custom_comparison, 4)
    )

    def test_init(self) -> None:
        assert self.shell_normal._data == [1, 2, 3, 4]
        assert self.shell_normal.cpd_core.scrubber.data == [1, 2, 3, 4]
        assert isinstance(self.shell_normal.cpd_core.algorithm, GraphAlgorithm)

        assert isinstance(self.shell_default.cpd_core.algorithm, GraphAlgorithm)
        assert isinstance(self.shell_default.cpd_core.scrubber, Scrubber)

        assert isinstance(self.shell_marked_data._data, LabeledCPData)

        assert self.shell_marked_data._data.raw_data == [1, 2, 3, 4]
        assert self.shell_marked_data._data.expected_res == [4, 5, 6, 7]
        assert list(self.shell_marked_data.scrubber.data.__iter__()) == [1, 2, 3, 4]

    def test_data_getter_setter(self) -> None:
        assert True

    def test_scrubber_getter_setter(self) -> None:
        assert True

    def test_CPDalgorithm_getter_setter(self) -> None:
        assert True

    def test_scenario_getter_setter(self) -> None:
        assert True

    def test_run_CPD(self) -> None:
        assert self.shell_normal.run_cpd() == {"result": []}
        assert self.shell_default.run_cpd() == {"result": []}
        assert self.shell_marked_data.run_cpd() == {"result": [], "expected": [4, 5, 6, 7]}
