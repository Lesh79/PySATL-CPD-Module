"""
Mock online algorithm that returns a sequence of states.
"""

from pysatl_cpd._typing import Number
from tests.mocks.online.algorithms.base import MockAlgorithmState
from tests.mocks.online.algorithms.simple import MockOnlineAlgorithm


class MockOnlineAlgorithmWithStateSequence(MockOnlineAlgorithm):
    """
    Mock algorithm that returns custom state snapshots.

    This algorithm cycles through a predefined sequence of states
    each time the state property is accessed.

    Parameters
    ----------
    state_sequence : list[MockAlgorithmState]
        Sequence of states to return on consecutive state property accesses.
    name : str, default="StateSequenceAlgorithm"
        Algorithm name.
    learning_period_size : int, default=0
        Learning period size.
    process_return_sequence : list[Number] | None, default=None
        Sequence of return values for process() calls. If provided, cycles
        through values; if None, returns 0.0 for all calls.
    expose_state : bool, default=True
        Whether to expose state.
    """

    def __init__(
        self,
        state_sequence: list[MockAlgorithmState],
        name: str = "StateSequenceAlgorithm",
        learning_period_size: int = 0,
        process_return_sequence: list[Number] | None = None,
        expose_state: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            learning_period_size=learning_period_size,
            process_return_value=0.0,
            expose_state=expose_state,
            process_return_sequence=process_return_sequence,
        )
        self._state_sequence = state_sequence
        self._state_index = 0

    @property
    def state(self) -> MockAlgorithmState | None:
        """Return the next state from sequence."""
        if self._state_index < len(self._state_sequence):
            state = self._state_sequence[self._state_index]
            self._state_index += 1
            return state
        return None

    def get_state_sequence_index(self) -> int:
        """Return the current index in state sequence."""
        return self._state_index

    def reset_state_sequence(self) -> None:
        """Reset the state sequence index to beginning."""
        self._state_index = 0
