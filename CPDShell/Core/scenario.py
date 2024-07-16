from dataclasses import dataclass


@dataclass
class Scenario:
    """Scenario for scrubber

    :param change_point_number: how many change points need to be found, defaults to 1
    :type change_point_number: int
    :param to_localize: is it necessary to localize change points, defaults to False
    :type to_localize: bool
    """

    change_point_number: int = 1
    to_localize: bool = False
