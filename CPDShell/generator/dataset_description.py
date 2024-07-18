from io import StringIO
from itertools import accumulate
from typing import Generic, TypeVar

from .distributions import Distribution

D = TypeVar("D", bound=Distribution)


class SampleDescription(Generic[D]):
    """Contains dataset description:

    * sub-samples lengths;
    * sub-samples distributions.

    Also can represent it in AsciiDoc format.
    """

    _name: str
    _samples_length: list[int]
    _samples_distributions: list[D]

    def __init__(
        self,
        name: str,
        samples_length: list[int],
        samples_distributions: list[D],
    ) -> None:
        """
        Creates new DatasetDescription instance.

        :param name: Name for the sample.
        :param samples_length: List of sub-samples length.
        :param samples_distributions: List of sub-samples distributions.
        """
        self._name = name
        self._samples_length = samples_length
        self._samples_distributions = samples_distributions
        assert len(self._samples_length) == len(self._samples_distributions)

    @property
    def name(self) -> str:
        return self._name

    @property
    def changepoints(self) -> list[int]:
        return list(accumulate(self._samples_length))[:-1]

    @property
    def length(self) -> list[int]:
        return self._samples_length

    @property
    def distributions(self) -> list[D]:
        return self._samples_distributions

    def to_asciidoc(self, image_path: str | None = None) -> str:
        """
        Converts `DatasetDescription` instance to string in AsciiDoc format.
        This description contain information about sample length, sub-samples lengths and distributions,
        changepoints indices in sample.

        Example
        -------
        .. code-block::

            = Sample 20-normal-0-1-20-normal-10-1

            [horizontal]
            Sample length:: 40
            Sub-samples lengths:: [20, 20]
            Change points:: [20]

            == Distributions

            . normal
            [horizontal]
            mean:: 0.0
            variance:: 1.0
            . normal
            [horizontal]
            mean:: 10.0
            variance:: 1.0

        :return: Dataset description string in AsciiDoc format.
        """
        description = StringIO()
        description.write(f"= Sample {self._name}\n\n")
        description.write("[horizontal]\n")
        description.write(f"Sample length:: {sum(self._samples_length)}\n")
        description.write(f"Subsamples lengths:: {self._samples_length}\n")
        description.write(f"Change points:: {self.changepoints}\n\n")
        description.write("== Distributions\n\n")
        for i in range(len(self._samples_length)):
            distr = self._samples_distributions[i]
            description.write(f". {distr.name}\n")
            description.write("[horizontal]\n")
            for k, v in distr.params.items():
                description.write(f"{k}:: {v}\n")

        if image_path:
            description.write("\n")
            description.write(f"image::{image_path}[Sample]\n")

        return description.getvalue()


class DatasetDescriptionBuilder:
    """Builder for `DatasetDescription` instance."""

    _name: str | None
    _samples_length: list[int] | None
    _samples_distributions: list[Distribution] | None

    def __init__(self):
        """Creates new DatasetDescriptionBuilder empty instance."""
        self._name = None
        self._samples_length = None
        self._samples_distributions = None

    def set_name(self, name: str) -> None:
        self._name = name

    def set_samples_lengths(self, samples_lengths: list[int]) -> None:
        self._samples_length = samples_lengths

    def set_samples_distributions(self, samples_distributions: list[Distribution]) -> None:
        self._samples_distributions = samples_distributions

    def build(self) -> SampleDescription:
        """
        Validate parameters and create `DatasetDescription` instance.

        :return: New `DatasetDescription` instance.
        """
        assert self._name
        assert self._samples_length
        assert self._samples_distributions
        assert len(self._samples_length) == len(self._samples_distributions)
        return SampleDescription(self._name, self._samples_length, self._samples_distributions)
