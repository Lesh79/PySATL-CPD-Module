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

from CPDShell.Core.algorithms.BayesianCPD.abstracts.ilocalizer import ILocalizer


class SimpleLocalizer(ILocalizer):
    """
    A localizer that localizes a change point corresponding with the most probable non-max run length.
    """

    def localize(self, growth_probs: np.ndarray) -> int:
        """
        Localizes a change point corresponding with the most probable non-max run length.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: the most probable non-max run length corresponding change point.
        """
        if len(growth_probs) == 0:
            return 0

        return int(np.argmax(growth_probs[0 : len(growth_probs) - 1]))
