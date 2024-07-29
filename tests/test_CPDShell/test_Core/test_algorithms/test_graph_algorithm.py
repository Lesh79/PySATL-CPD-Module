import pytest

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm


def custom_comparison(node1, node2):
    arg = 5
    return abs(node1 - node2) <= arg


class TestGraphAlgorithm:
    @pytest.mark.parametrize(
        "alg_param,data,expected",
        (((custom_comparison, 1.5), (50, 55, 60, 48, 52, 70, 75, 80, 90, 85, 95, 100, 50), [5]),),
    )
    def test_localize(self, alg_param, data, expected):
        algorithm = GraphAlgorithm(*alg_param)
        assert algorithm.localize(data) == expected

    @pytest.mark.parametrize(
        "alg_param,data,expected",
        (((custom_comparison, 1.5), (50, 55, 60, 48, 52, 70, 75, 80, 90, 85, 95, 100, 50), 1),),
    )
    def test_detect(self, alg_param, data, expected):
        algorithm = GraphAlgorithm(*alg_param)
        assert algorithm.detect(data) == expected
