"""
MIT License

Copyright (c) 2024 Alexey Tatyanenko

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

from CPDShell.Core.algorithms.BayesianCPD.abstracts.ihazard import IHazard


class ConstantHazard(IHazard):
    """
    A constant hazard function, corresponding to an exponential distribution with a given rate.
    """

    def __init__(self, rate: float):
        """
        Initializes the constant hazard function with a given rate of an underlying exponential distribution.
        :param rate: rate of an underlying exponential distribution.
        """
        self.__rate = rate

    def hazard(self, run_lengths: np.ndarray) -> np.ndarray:
        """
        Calculates the constant hazard function.
        :param run_lengths: run lengths at the time.
        :return: hazard function's values for given run lengths.
        """
        return np.ones(len(run_lengths)) / self.__rate
