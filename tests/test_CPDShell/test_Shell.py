from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.Shell import CPDShell, MarkedCPData


class TestMarkedCPData:
    data = MarkedCPData([1, 2, 3], [4, 5, 6])

    def test_init(self) -> None:
        assert self.data.raw_data == [1, 2, 3]
        assert self.data.expected_res == [4, 5, 6]

    def test_iter(self) -> None:
        assert list(self.data.__iter__()) == [1, 2, 3]

    def test_generate_CP_dataset(self) -> None:
        generated = MarkedCPData.generate_CP_dataset("shish")
        assert generated.raw_data == ["chin chon"]
        assert generated.expected_res == ["gop", "stop"]


class TestCPDShell:
    shell_normal = CPDShell([1, 2, 3], algorithm=GraphAlgorithm(3, 4), scrubber_class=Scrubber)
    shell_default = CPDShell([3, 4, 5])
    shell_marked_data = CPDShell(MarkedCPData([1, 2, 3], [4, 5, 6]))

    def test_init(self) -> None:
        assert self.shell_normal._data == [1, 2, 3]
        assert self.shell_normal.cpd_core.scrubber.data == [1, 2, 3]
        assert isinstance(self.shell_normal.cpd_core.algorithm, GraphAlgorithm)

        assert isinstance(self.shell_default.cpd_core.algorithm, GraphAlgorithm)
        assert isinstance(self.shell_default.cpd_core.scrubber, Scrubber)

        assert self.shell_marked_data._data.raw_data == [1, 2, 3]
        assert self.shell_marked_data._data.expected_res == [4, 5, 6]
        assert list(self.shell_marked_data.scrubber.data.__iter__()) == [1, 2, 3]

    def test_data_getter_setter(self) -> None:
        assert True

    def test_scrubber_getter_setter(self) -> None:
        assert True

    def test_CPDalgorithm_getter_setter(self) -> None:
        assert True

    def test_scenario_getter_setter(self) -> None:
        assert True

    def test_run_CPD(self) -> None:
        assert self.shell_normal.run_CPD() == {"result": [0]}
        assert self.shell_default.run_CPD() == {"result": [0]}
        assert self.shell_marked_data.run_CPD() == {"result": [0], "expected": [4, 5, 6]}
