from dataclasses import dataclass


@dataclass
class Scenario:
    """Scenario for scrubber

    :param change_point_number: how many change points need to be found, defaults to 1
    :param to_localize: is it necessary to localize change points, defaults to False
    """

    change_point_number: int = 1
    to_localize: bool = False
