"""
Mock online algorithm that returns a sequence of values.
"""

from pysatl_cpd._typing import Number
from tests.mocks.online.algorithms.simple import MockOnlineAlgorithm


class MockOnlineAlgorithmWithSequence(MockOnlineAlgorithm):
    """
    Mock algorithm that returns a sequence of values for process() calls.

    This is a convenience subclass for testing with predefined response sequences.

    Parameters
    ----------
    return_sequence : list[Number]
        Sequence of return values for consecutive process() calls.
    name : str, default="SequenceAlgorithm"
        Algorithm name.
    learning_period_size : int, default=0
        Learning period size.
    expose_state : bool, default=True
        Whether to expose state.
    """

    def __init__(
        self,
        return_sequence: list[Number],
        name: str = "SequenceAlgorithm",
        learning_period_size: int = 0,
        expose_state: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            learning_period_size=learning_period_size,
            process_return_value=0.0,
            expose_state=expose_state,
            process_return_sequence=return_sequence,
        )
