import typing as tp
from collections import deque
from itertools import islice

from observations.observation import Observation, Observations
from observations.observation_heap import NNHeap


class KNNGraph:
    def __init__(
        self, observations_count: int, observations: Observations, metric: tp.Callable[[Observation, Observation], float], k=3
    ) -> None:
        self._window_size = observations_count
        self._observations = observations
        self._window = deque(islice(self._observations,
                                    len(self._observations) - self._window_size,
                                    len(self._observations)), maxlen=self._window_size)
        self._metric = metric
        self._graph: deque[NNHeap] = deque(maxlen=self._window_size)
        self._k = k

    def build(self) -> None:
        """
        Build KNN graph according to the given parameters
        """
        for i in range(self._window_size):
            heap = NNHeap(self._k, self._metric, self._observations[-i - 1])
            heap.build(self._window)
            self._graph.appendleft(heap)

    def update(self, observation: Observation) -> None:
        """
        Add observation to KNN graph
        :param observation: New observation
        """
        obsolete_obs = self._window[0]
        self._window.append(observation)
        self._graph.popleft()

        for heap in self._graph:
            heap.remove(obsolete_obs, self._window)
            heap.add(observation)

        new_heap = NNHeap(self._k, self._metric, observation)
        new_heap.build(self._window)
        self._graph.append(new_heap)

    def check_neighbour(self, first_index: int, second_index: int) -> bool:
        """
        Checks if the second observation is among the k nearest neighbours of the first observation
        :param first_index:
        :param second_index:
        :return:
        """
        neighbour = self._window[second_index]
        return self._graph[first_index].find_in_heap(neighbour)
