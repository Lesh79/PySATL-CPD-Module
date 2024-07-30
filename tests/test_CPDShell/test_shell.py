import tempfile
from os import walk
from pathlib import Path

import pytest

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.scenario import Scenario
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.shell import CPContainer, CPDShell, LabeledCPData


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
    shell_for_setter_getter = CPDShell(
        [4, 3, 2, 1], cpd_algorithm=GraphAlgorithm(custom_comparison, 4), scrubber_class=Scrubber
    )
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
        assert self.shell_for_setter_getter.data == [4, 3, 2, 1]
        assert self.shell_for_setter_getter.cpd_core.scrubber.data == [4, 3, 2, 1]

        self.shell_for_setter_getter.data = [1, 3, 4]

        assert self.shell_for_setter_getter.data == [1, 3, 4]
        assert self.shell_for_setter_getter.cpd_core.scrubber.data == [1, 3, 4]

    def test_scrubber_setter(self) -> None:
        class TestNewScrubber(Scrubber):
            pass

        previous_scrubber = self.shell_for_setter_getter.scrubber
        self.shell_for_setter_getter.scrubber = TestNewScrubber
        assert isinstance(self.shell_for_setter_getter.scrubber, TestNewScrubber)
        assert self.shell_for_setter_getter.scrubber.data == previous_scrubber.data
        assert self.shell_for_setter_getter.scrubber.scenario == previous_scrubber.scenario

    def test_CPDalgorithm_getter_setter(self) -> None:
        FIVE = 5

        class TestNewAlgo(GraphAlgorithm):
            pass

        self.shell_for_setter_getter.CPDalgorithm = TestNewAlgo(custom_comparison, 5)
        assert isinstance(self.shell_for_setter_getter.cpd_core.algorithm, TestNewAlgo)
        assert self.shell_for_setter_getter.cpd_core.algorithm.threshold == FIVE

    def test_scenario_getter_setter(self) -> None:
        self.shell_for_setter_getter.scenario = Scenario(20, False)
        assert self.shell_for_setter_getter.cpd_core.scrubber.scenario == Scenario(20, False)

    def test_change_scenario(self) -> None:
        self.shell_for_setter_getter.change_scenario(15, True)
        assert self.shell_for_setter_getter.scenario == Scenario(15, True)

    def test_run_CPD(self) -> None:
        res_normal = self.shell_normal.run_cpd()
        res_def = self.shell_default.run_cpd()
        res_marked = self.shell_marked_data.run_cpd()
        assert res_normal.result == []
        assert res_normal.expected_result is None

        assert res_def.result == []
        assert res_def.expected_result is None

        assert res_marked.result == []
        assert res_marked.expected_result == [4, 5, 6, 7]
        assert res_marked.result_diff == [4, 5, 6, 7]


class TestCPContainer:
    cont_default1 = CPContainer([1, 2, 3], [2, 3, 4], 10)
    cont_default2 = CPContainer([1, 2, 3, 6, 8], [2, 3, 4, 6], 20)
    cont_no_expected = CPContainer([1, 2, 3], None, 5)

    def test_result_diff(self) -> None:
        assert self.cont_default1.result_diff == [1, 4]
        assert self.cont_default2.result_diff == [1, 4, 8]

    def test_result_diff_execption_case(self) -> None:
        with pytest.raises(ValueError):
            self.cont_no_expected.result_diff

    def test_str_cpcontainer(self) -> None:
        assert (
            str(self.cont_default1)
            == """Located change points: (1;2;3)
Expected change point: (2;3;4)
Difference: (1;4)
Computation time (ms): 10"""
        )

        assert (
            str(self.cont_default2)
            == """Located change points: (1;2;3;6;8)
Expected change point: (2;3;4;6)
Difference: (1;4;8)
Computation time (ms): 20"""
        )

        assert (
            str(self.cont_no_expected)
            == """Located change points: (1;2;3)
Computation time (ms): 5"""
        )
