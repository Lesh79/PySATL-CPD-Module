import numpy as np 
from dataclasses import dataclass, field
from collections import deque
from typing import TypeAlias


@dataclass(order=True)
class Observation:
    time: int
    value: float | np.float64 = field(compare=False)


@dataclass(order=True)
class Neighbour:
    distance: float
    observation: Observation
