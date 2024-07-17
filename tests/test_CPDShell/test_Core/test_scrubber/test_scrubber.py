import pytest

from CPDShell.Core.scrubber.scrubber import *


class TestScrubber:

    @pytest.mark.parametrize(
        "scenario_param,data,expected_windows",
        (
            (
                (1, True),
                (1, 2, 3, 4, 5, 6, 7),
                [
                    (1, 2, 3, 4, 5, 6, 7),
                ],
            ),
        ),
    )
    def test_generate_window(self, scenario_param, data, expected_windows):
        scenario = Scenario(*scenario_param)
        scrubber = Scrubber(scenario, data)
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
        scrubber = Scrubber(scenario, data)
        while scrubber.is_running:
            scrubber.generate_window()
            scrubber.add_change_points(change_points.pop(0))
        assert scrubber.change_points == expected_change_points
