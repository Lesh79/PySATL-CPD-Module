import os.path
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Generic, TypeVar

import yaml

from .dataset_description import DatasetDescriptionBuilder, SampleDescription
from .distributions import Distribution

D = TypeVar("D", bound=Distribution)


class ConfigParser(Generic[D]):
    """
    Parse YAML generation config and provides an iterator over DatasetDescriptions.
    """

    NAME_FIELD: Final[str] = "name"
    DISTRIBUTIONS_FIELD: Final[str] = "distributions"
    LENGTHS_FIELD: Final[str] = "lengths"
    PARAMETERS_FIELD: Final[str] = "parameters"

    _descriptions: list[SampleDescription[D]]

    def __init__(self, config_path: Path):
        self.validate_config(config_path)
        with open(config_path) as cf:
            config: list[dict] = yaml.safe_load(cf)
            self._descriptions = self._parse_config(config)

    def __iter__(self) -> Iterator[SampleDescription[D]]:
        return self._descriptions.__iter__()

    @staticmethod
    def _parse_config(config: list[dict]) -> list[SampleDescription[D]]:
        descriptions: list[SampleDescription[D]] = []
        for descr in config:
            db = DatasetDescriptionBuilder()
            db.set_name(descr[ConfigParser.NAME_FIELD])
            db.set_samples_lengths(descr[ConfigParser.LENGTHS_FIELD])
            distrs = descr[ConfigParser.DISTRIBUTIONS_FIELD]
            params = descr[ConfigParser.PARAMETERS_FIELD]
            db.set_samples_distributions([Distribution.from_str(distr, prms) for distr, prms in zip(distrs, params)])
            descriptions.append(db.build())
        return descriptions

    @staticmethod
    def validate_config(config_path: Path) -> None:
        """
        Reads and validates samples generation config file.
        Raise `TypeError` or `ValueError` if config is not valid.

        :param config_path: Path to configuration file.
        """
        if not os.path.exists(config_path):
            raise ValueError("Incorrect config path")
        with open(config_path) as cf:
            config = yaml.safe_load(cf)
        if not isinstance(config, list):
            raise TypeError("Config must be a list of dataset descriptions")
        for i, descr in enumerate(config):
            if not isinstance(descr, dict):
                raise TypeError(f"Description #{i} is not a dictionary")
            name = descr.get(ConfigParser.NAME_FIELD, None)
            ConfigParser._validate_description_name(i, name)
            lengths: list | None = descr.get(ConfigParser.LENGTHS_FIELD, None)
            ConfigParser._validate_description_list_field(i, lengths, ConfigParser.LENGTHS_FIELD, int)
            distributions: list | None = descr.get(ConfigParser.DISTRIBUTIONS_FIELD, None)
            ConfigParser._validate_description_list_field(i, distributions, ConfigParser.DISTRIBUTIONS_FIELD, str)
            params: list | None = descr.get(ConfigParser.PARAMETERS_FIELD, None)
            ConfigParser._validate_description_list_field(i, params, ConfigParser.PARAMETERS_FIELD, dict)
            if (
                lengths is not None
                and distributions is not None
                and params is not None
                and not (len(lengths) == len(distributions) == len(params))
            ):
                raise ValueError(
                    f"Description #{i} size of {ConfigParser.LENGTHS_FIELD}, {ConfigParser.DISTRIBUTIONS_FIELD} and "
                    + f"{ConfigParser.PARAMETERS_FIELD} are not equal"
                )
            if distributions is not None and params is not None:
                for distr, prms in zip(distributions, params):
                    try:
                        Distribution.from_str(distr, prms)
                    except Exception as e:
                        raise ValueError(f"Description #{i} distribution is invalid") from e
            else:
                raise ValueError("distributions and params can not be None")

    @staticmethod
    def _validate_description_name(index: int, name):
        if not name:
            raise ValueError(f"Description #{index} does not contain a name")
        if not isinstance(name, str):
            raise TypeError(f"Description #{index} name is not string")

    @staticmethod
    def _validate_description_list_field(index: int, list_field, list_field_name: str, elements_type: type):
        if not list_field:
            raise ValueError(f"Description #{index} does not contain a {list_field_name} list for sub-samples")
        if not isinstance(list_field, list):
            raise TypeError(f"Description #{index} {list_field_name} is not list")
        for l_idx, length in enumerate(list_field):
            if not isinstance(length, elements_type):
                raise TypeError(f"Description #{index} lengths[{l_idx}] is not {elements_type}")
