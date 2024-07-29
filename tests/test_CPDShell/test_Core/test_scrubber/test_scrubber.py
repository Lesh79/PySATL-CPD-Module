import pytest

from CPDShell.Core.scrubber.scrubber import Scenario, Scrubber


class TestScrubber:
    @pytest.mark.parametrize(
        "scenario_param,data,window_length,expected_windows",
        (
            (
                (1, True),
                (1, 2, 3, 4, 5, 6, 7),
                5,
                [(1, 2, 3, 4, 5), (2, 3, 4, 5, 6)],
            ),
            (
                (1, True),
                (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                7,
                [(1, 2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7, 8, 9), (5, 6, 7, 8, 9, 10, 11)],
            ),
        ),
    )
    def test_generate_window(self, scenario_param, data, window_length, expected_windows):
        scenario = Scenario(*scenario_param)
        scrubber = Scrubber(scenario, data, window_length)
        while scrubber.is_running:
            assert scrubber.generate_window() == expected_windows.pop(0)
            scrubber.add_change_points([])

    @pytest.mark.parametrize(
        "scenario_param,data,change_points,expected_change_points",
        (
            (
                (1, True),
                (1, 2, 3, 4, 5, 6, 7),
                [
                    [1, 2],
                ],
                [1],
            ),
            (
                (2, True),
                (1, 2, 3, 4, 5, 6, 7),
                [
                    [1, 2],
                ],
                [1, 2],
            ),
        ),
    )
    def test_add_change_points(self, scenario_param, data, change_points, expected_change_points):
        scenario = Scenario(*scenario_param)
        scrubber = Scrubber(scenario, data, 100)
        while scrubber.is_running:
            scrubber.generate_window()
            scrubber.add_change_points(change_points.pop(0))
        assert scrubber.change_points == expected_change_points
