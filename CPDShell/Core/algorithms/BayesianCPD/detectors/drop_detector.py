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

from CPDShell.Core.algorithms.BayesianCPD.abstracts.idetector import IDetector


class DropDetector(IDetector):
    """
    A detector that detects a change point if the instantaneous drop in the probability of the maximum run length
    exceeds the threshold.
    """

    def __init__(self, threshold: float):
        """
        Initializes the detector with given drop threshold.
        :param threshold: threshold for a drop of the maximum run length's probability.
        """
        self.__previous_growth_prob: None | float = None
        self.__threshold = threshold
        assert 0.0 <= self.__threshold <= 1.0

    def detect(self, growth_probs: np.ndarray) -> bool:
        """
        Checks whether a changepoint occurred with given growth probabilities at the time.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: boolean indicating whether a changepoint occurred.
        """
        if len(growth_probs) == 0:
            return False

        last_growth_prob = growth_probs[len(growth_probs) - 1]
        if self.__previous_growth_prob is None:
            self.__previous_growth_prob = last_growth_prob
            return False

        drop = float(self.__previous_growth_prob - last_growth_prob)
        assert drop >= 0.0

        return drop >= self.__threshold

    def clear(self) -> None:
        """
        Clears the detector's state.
        :return:
        """
        self.__previous_growth_prob = None
