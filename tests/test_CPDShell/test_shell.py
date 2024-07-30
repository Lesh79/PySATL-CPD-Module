from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.scrubber.scrubber import Scrubber
from CPDShell.shell import CPContainer, CPDShell, LabeledCPData


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

        # assert isinstance(self.shell_marked_data._data, LabeledCPData)

        # assert self.shell_marked_data._data.raw_data == [1, 2, 3, 4]
        # assert self.shell_marked_data._data.expected_res == [4, 5, 6, 7]
        # assert list(self.shell_marked_data.scrubber.data.__iter__()) == [1, 2, 3, 4]

    def test_data_getter_setter(self) -> None:
        assert True

    def test_scrubber_getter_setter(self) -> None:
        assert True

    def test_CPDalgorithm_getter_setter(self) -> None:
        assert True

    def test_scenario_getter_setter(self) -> None:
        assert True

    def test_run_CPD(self) -> None:
        assert self.shell_normal.run_cpd() == CPContainer([], None)
        assert self.shell_default.run_cpd() == CPContainer([], None)
        assert self.shell_marked_data.run_cpd() == CPContainer([], [4, 5, 6, 7])
