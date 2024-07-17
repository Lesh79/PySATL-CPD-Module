import pytest

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.CPDCore import CPDCore, Scrubber
from CPDShell.Core.scenario import Scenario


class TestCPDCore:
    @pytest.mark.parametrize(
        "scenario_param,data,alg_class,alg_param,expected",
        (
            ((1, True), (1, 2, 3, 4, 5, 6, 7), GraphAlgorithm, (1, 2), [0]),
            ((1, False), (1, 2, 3, 4, 5, 6, 7), GraphAlgorithm, (1, 2), [10]),
        ),
    )
    def test_run(self, scenario_param, data, alg_class, alg_param, expected):
        scenario = Scenario(*scenario_param)
        scrubber = Scrubber(scenario, data)
        algorithm = alg_class(*alg_param)

        core = CPDCore(scrubber, algorithm)
        assert core.run() == expected
