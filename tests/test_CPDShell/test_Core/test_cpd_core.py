import pytest

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.cpd_core import CPDCore, Scrubber
from CPDShell.Core.scenario import Scenario


def custom_comparison(node1, node2):
    arg = 1
    return abs(node1 - node2) <= arg


class TestCPDCore:
    @pytest.mark.parametrize(
        "scenario_param,data,alg_class,alg_param,expected",
        (
            (
                (1, True),
                (1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100),
                GraphAlgorithm,
                (custom_comparison, 2),
                [6],
            ),
            (
                (1, False),
                (1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100, 100, 100),
                GraphAlgorithm,
                (custom_comparison, 2),
                [10],
            ),
        ),
    )
    def test_run(self, scenario_param, data, alg_class, alg_param, expected):
        scenario = Scenario(*scenario_param)
        scrubber = Scrubber(scenario, data)
        algorithm = alg_class(*alg_param)

        core = CPDCore(scrubber, algorithm)
        assert core.run() == expected
