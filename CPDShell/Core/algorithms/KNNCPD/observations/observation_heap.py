import heapq
import typing as tp

from observations.observation import Observation, Neighbour, Observations


class NNHeap:
    def __init__(
        self, size: int, metric: tp.Callable[[Observation, Observation], float], main_observation: Observation
    ) -> None:
        self._size = size
        self._metric = metric
        self._main_observation = main_observation
        self._heap: list[Neighbour] = []
        self._auxiliary_heap: list[Neighbour] = []

    def build(self, neighbours: Observations) -> None:
        for neighbour in neighbours:
            self.add(neighbour)

    def add(self, observation: Observation) -> None:
        if observation is self._main_observation:
            return

        neg_distance = -self._metric(self._main_observation, observation)
        neighbour = Neighbour(neg_distance, observation)

        if len(self._heap) == self._size and neighbour.distance >= self._heap[0].distance:
            old_neighbour = heapq.heapreplace(self._heap, neighbour)
            self.add_auxiliary(old_neighbour)
        elif len(self._heap) == self._size and neighbour.distance < self._heap[0].distance:
            self.add_auxiliary(neighbour)
        else:
            heapq.heappush(self._heap, neighbour)

    def add_auxiliary(self, neighbour: Neighbour) -> None:
        if len(self._auxiliary_heap) == self._size and neighbour.distance >= self._auxiliary_heap[0].distance:
            heapq.heapreplace(self._auxiliary_heap, neighbour)
        elif len(self._auxiliary_heap) < self._size:
            heapq.heappush(self._auxiliary_heap, neighbour)

    def remove(self, observation: Observation, observations: Observations) -> None:
        if not self._heap:
            return

        neg_distance = -self._metric(self._main_observation, observation)
        neighbour = Neighbour(neg_distance, observation)

        if neg_distance >= self._heap[0].distance and neighbour in self._heap:
            self._heap.remove(neighbour)
            heapq.heapify(self._heap)

            if self._auxiliary_heap:
                new_neighbour = heapq.nlargest(1, self._auxiliary_heap)[0]
                self._auxiliary_heap.remove(new_neighbour)
                heapq.heapify(self._auxiliary_heap)
                heapq.heappush(self._heap, new_neighbour)
            else:
                self.build(observations)
        elif (self._auxiliary_heap
              and neg_distance >= self._auxiliary_heap[0].distance
              and neighbour in self._auxiliary_heap):
            self._auxiliary_heap.remove(neighbour)
            heapq.heapify(self._auxiliary_heap)

    def find_in_heap(self, observation: Observation) -> bool:
        def predicate(x: Neighbour) -> bool:
            return x.observation is observation

        return any(predicate(i) for i in self._heap)
