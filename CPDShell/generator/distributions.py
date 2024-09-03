from enum import Enum
from typing import Final, Protocol

import numpy as np
import scipy.stats as ss


class Distributions(Enum):
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    WEIBULL = "weibull"
    UNIFORM = "uniform"
    BETA = "beta"

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
            case Distributions.WEIBULL.value:
                return WeibullDistribution.from_params(params)
            case Distributions.UNIFORM.value:
                return UniformDistribution.from_params(params)
            case Distributions.BETA.value:
                return BetaDistribution.from_params(params)
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
                "Exponential distribution must have 1 parameters: " + f"{ExponentialDistribution.RATE_KEY}"
            )
        rate: float = float(params[ExponentialDistribution.RATE_KEY])
        if rate <= 0:
            raise ValueError("Rate must be greater than 0")
        return ExponentialDistribution(rate)


class WeibullDistribution(ScipyDistribution):
    """
    Description of weibull distribution with intensity parameter.
    """

    SHAPE_KEY: Final[str] = "shape"
    SCALE_KEY: Final[str] = "scale"

    shape: float
    scale: float

    def __init__(self, shape: float = 1.0, scale: float = 1.0):
        if shape <= 0 or scale <= 0:
            raise ValueError("Shape and scale must be greater than 0")
        self.shape = shape
        self.scale = scale

    @property
    def name(self) -> str:
        return str(Distributions.WEIBULL)

    @property
    def params(self) -> dict[str, str]:
        return {
            WeibullDistribution.SHAPE_KEY: str(self.shape),
            WeibullDistribution.SCALE_KEY: str(self.scale),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.weibull_min(c=self.shape, scale=1 / self.scale).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "WeibullDistribution":
        num_params = 2
        if len(params) != num_params:
            raise ValueError(
                "Exponential distribution must have 2 parameters: "
                + f"{WeibullDistribution.SHAPE_KEY}"
                + f"{WeibullDistribution.SCALE_KEY}"
            )
        shape: float = float(params[WeibullDistribution.SHAPE_KEY])
        scale: float = float(params[WeibullDistribution.SCALE_KEY])
        if shape <= 0 or scale <= 0:
            raise ValueError("Parameters must be greater than 0")
        return WeibullDistribution(shape, scale)


class UniformDistribution(ScipyDistribution):
    """
    Description of uniform distribution with intensity parameter.
    """

    MIN_KEY: Final[str] = "min"
    MAX_KEY: Final[str] = "max"

    max: float
    min: float

    def __init__(self, min_value: float, max_value: float):
        if min_value >= max_value:
            raise ValueError("Max must be greater than min value")
        self.min = min_value
        self.max = max_value

    @property
    def name(self) -> str:
        return str(Distributions.UNIFORM)

    @property
    def params(self) -> dict[str, str]:
        return {
            UniformDistribution.MIN_KEY: str(self.min),
            UniformDistribution.MAX_KEY: str(self.max),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.uniform(loc=self.min, scale=self.max - self.min).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "UniformDistribution":
        num_params = 2
        if len(params) != num_params:
            raise ValueError(
                "Uniform distribution must have 2 parameters: "
                + f"{UniformDistribution.min}"
                + f"{UniformDistribution.max}"
            )
        min_value: float = float(params[UniformDistribution.MIN_KEY])
        max_value: float = float(params[UniformDistribution.MAX_KEY])
        if min_value >= max_value:
            raise "Max must be greater than min value"
        return UniformDistribution(min_value, max_value)


class BetaDistribution(ScipyDistribution):
    """
    Description of beta distribution with intensity parameter.
    """

    ALPHA_KEY: Final[str] = "alpha"
    BETA_KEY: Final[str] = "beta"

    alpha: float
    beta: float

    def __init__(self, alpha_value: float, beta_value: float):
        if alpha_value <= 0 or beta_value <= 0:
            raise ValueError("Alpha and beta must be greater than zero")
        self.alpha = alpha_value
        self.beta = beta_value

    @property
    def name(self) -> str:
        return str(Distributions.BETA)

    @property
    def params(self) -> dict[str, str]:
        return {
            BetaDistribution.ALPHA_KEY: str(self.alpha),
            BetaDistribution.BETA_KEY: str(self.beta),
        }

    def scipy_sample(self, length: int) -> np.ndarray:
        return ss.beta(a=self.alpha, b=self.beta).rvs(size=length)

    @staticmethod
    def from_params(params: dict[str, str]) -> "BetaDistribution":
        num_params = 2
        if len(params) != num_params:
            raise ValueError(
                f"Beta distribution must have 2 parameters: {BetaDistribution.ALPHA_KEY}, {BetaDistribution.BETA_KEY}"
            )
        alpha: float = float(params[BetaDistribution.ALPHA_KEY])
        beta: float = float(params[BetaDistribution.BETA_KEY])
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be greater than zero")
        return BetaDistribution(alpha, beta)
