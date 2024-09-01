from enum import Enum
from typing import Final, Protocol

import numpy as np
import scipy.stats as ss


class Distributions(Enum):
    NORMAL = "normal"
    EXPONENTIAL = "exponential"

    def __str__(self):
        return self.value


class Distribution(Protocol):
    """
    An interface for all distributions.
    Allows to create instance with name and params dictionary.
    """

    @property
    def name(self) -> str:
        """
        :return: Name of the distribution.
        """
        return ""

    @property
    def params(self) -> dict[str, str]:
        """
        :return: Parameters of the distribution.
        """
        return {}

    @staticmethod
    def from_str(name: str, params: dict[str, str]) -> "Distribution":
        match name:
            case Distributions.NORMAL.value:
                return NormalDistribution.from_params(params)
            case Distributions.EXPONENTIAL.value:
                return ExponentialDistribution.from_params(params)
            case _:
                raise NotImplementedError()


class ScipyDistribution(Distribution):
    """
    Distribution supporting sample generation with SciPy methods.
    """

    def scipy_sample(self, length: int) -> np.ndarray:
        """
        Generate sample using SciPy.

        :return: Generated sample.
        """
        raise NotImplementedError()


class NormalDistribution(ScipyDistribution):
    """
    Description for the normal distribution with mean and variance parameters.
    """

    MEAN_KEY: Final[str] = "mean"
    VAR_KEY: Final[str] = "variance"

    mean: float
    variance: float

    def __init__(self, mean=0.0, var=1.0):
        self.mean = mean
        self.variance = var

    @property
    def name(self) -> str:
        return str(Distributions.NORMAL)

    @property
    def params(self) -> dict[str, str]:
        return {
            NormalDistribution.MEAN_KEY: str(self.mean),
            NormalDistribution.VAR_KEY: str(self.variance),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.norm.rvs(loc=self.mean, scale=self.variance, size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "NormalDistribution":
        parameter_number = 2
        if len(params) != parameter_number:
            raise ValueError(
                "Normal distribution must have 2 parameters: "
                + f"{NormalDistribution.MEAN_KEY}, {NormalDistribution.VAR_KEY}"
            )
        mean: float = float(params[NormalDistribution.MEAN_KEY])
        var: float = float(params[NormalDistribution.VAR_KEY])
        if var < 0:
            raise ValueError("Variance cannot be less than 0")
        return NormalDistribution(mean, var)


class ExponentialDistribution(ScipyDistribution):
    """
    Description of exponential distribution with intensity parameter.
    """
    RATE_KEY: Final[str] = "rate"

    rate: float

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError("Rate must be greater than 0")
        self.rate = rate

    @property
    def name(self) -> str:
        return str(Distributions.EXPONENTIAL)

    @property
    def params(self) -> dict[str, str]:
        return {
            ExponentialDistribution.RATE_KEY: str(self.rate),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.expon.rvs(scale=1 / self.rate, size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "ExponentialDistribution":
        if len(params) != 1:
            raise ValueError(
                "Exponential distribution must have 1 parameters: " + f"{ExponentialDistribution.RATE_KEY}")
        rate: float = float(params[ExponentialDistribution.RATE_KEY])
        if rate <= 0:
            raise ValueError("Rate must be greater than 0")
        return ExponentialDistribution(rate)
