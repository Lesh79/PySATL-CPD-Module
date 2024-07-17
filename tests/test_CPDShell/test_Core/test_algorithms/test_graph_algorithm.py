import pytest

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm


class TestGraphAlgorithm:

    @pytest.mark.parametrize("alg_param,data,expected", (((1, 2), (1, 2, 3, 4, 5, 6, 7), [0]),))
    def test_localize(self, alg_param, data, expected):
        algorithm = GraphAlgorithm(*alg_param)
        assert algorithm.localize(data) == expected

    @pytest.mark.parametrize("alg_param,data,expected", (((1, 2), (1, 2, 3, 4, 5, 6, 7), [10]),))
    def test_detect(self, alg_param, data, expected):
        algorithm = GraphAlgorithm(*alg_param)
        assert algorithm.detect(data) == expected
