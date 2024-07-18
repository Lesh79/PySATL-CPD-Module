from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from .dataset_description import SampleDescription


class DatasetSaver:
    """
    Saves samples and descriptions to specified directory.
    """

    SAMPLE_DATA: Final[str] = "sample.csv"
    DESCRIPTION: Final[str] = "sample.adoc"
    SAMPLE_IMAGE: Final[str] = "sample.png"
    CHANGEPOINTS_DATA: Final[str] = "changepoints.csv"

    _out_dir: Path
    _replace: bool

    def __init__(self, out_dir: Path, replace: bool):
        """
        :param out_dir: Directory to save samples and descriptions.
        :param replace: Whether sample should be saved if it already exists.
        """
        if not out_dir.exists():
            out_dir.mkdir()
        self._replace = replace
        self._out_dir = out_dir

    def save_sample(self, sample: np.ndarray, description: SampleDescription) -> bool:
        """
        Save sample, list of changepoints, sample plot and AsciiDoc description.

        :param sample: Sample to save.
        :param description: Description of the saving `sample`.
        :return: Whether sample and description have been saved to output directory.
        """
        sample_dir: Path = self._out_dir.joinpath(description.name)
        if sample_dir.exists() and not self._replace:
            return False
        if not sample_dir.exists():
            sample_dir.mkdir()
        # Save generated sample
        sample_file: Path = sample_dir.joinpath(DatasetSaver.SAMPLE_DATA)
        np.savetxt(sample_file, sample, delimiter=",")
        # Save changepoints list
        changepoints_file: Path = sample_dir.joinpath(DatasetSaver.CHANGEPOINTS_DATA)
        changepoints: list[int] = description.changepoints
        with open(changepoints_file, "w") as cf:
            for cp in changepoints:
                cf.write(f"{cp}\n")
        # Save sample plot
        image_file: Path = sample_dir.joinpath(DatasetSaver.SAMPLE_IMAGE)
        plt.plot(sample)
        plt.vlines(x=changepoints, ymin=sample.min(), ymax=sample.max(), colors="orange", ls="--")
        plt.savefig(image_file)
        plt.close()
        # Save description
        description_file: Path = sample_dir.joinpath(DatasetSaver.DESCRIPTION)
        with open(description_file, "w") as df:
            df.write(description.to_asciidoc(DatasetSaver.SAMPLE_IMAGE))

        return True
