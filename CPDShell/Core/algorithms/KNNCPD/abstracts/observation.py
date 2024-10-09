"""
Module for abstractions used in heap, needed to clearly distinguish observations made at different times.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field

import numpy as np


@dataclass(order=True)
class Observation:
    """
    Abstraction over observation that consists of the time of the point in time series and the value of it.
    """

    time: int
    value: float | np.float64 = field(compare=False)


@dataclass(order=True)
class Neighbour:
    """
    Abstraction over neighbour that consists of the distance to the main point and the observation-neighbour itself.
    """

    distance: float
    observation: Observation
