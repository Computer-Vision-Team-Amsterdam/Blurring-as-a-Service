import logging
import os
from enum import IntFlag
from functools import reduce
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel

from blurring_as_a_service.settings.attr_dict import AttrDict
from blurring_as_a_service.settings.settings_schema import (
    BlurringAsAServiceSettingsSpec,
)

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    pass


class SettingsMeta(type):
    """Metaclass for `GenericSettings` to allow easy access to the default settings."""

    def __getattr__(self, item):
        def_settings = self.get_settings()
        if def_settings:
            return def_settings[item]
        else:
            return super().__getattr__(item)

    def __getitem__(self, item):
        def_settings = self.get_settings()
        if def_settings:
            return def_settings[item]
        else:
            raise KeyError(f"key {item} does not exist")


class GenericSettings(AttrDict, metaclass=SettingsMeta):  # type: ignore
    """Generic configuration store. Identical to an `AttrDict` with some added i/o
    functionality.

    This class is "generic" in that it does not assume any specifics for the fraude
    models, but just loads settings from a file and provides singleton-like access
    to default settings.

    The default_settings can be registered via `register_default_settings()`. It accepts
    any mutable mapping. Once a dictionary is registered as default, it can be retrieved
    via the `get_default_settings()` method.

    For convenience, its elements are also directly accesible via item access directly
    on the `GenericSettings` class or as a class attribute (if it does not clash with
    any members of the `GenericSettings` class).

    The class also has a few convenience methods to load settings from yaml files
    (`from_yaml()`), or to load settings from yaml and register them as default in one
    go: `default_from_yaml()`.

    Examples
    --------
    >>> d = {'x': 42}
    >>> GenericSettings.set_settings(d)
    >>> assert GenericSettings.get_settings()['x'] == 42
    >>> assert GenericSettings['x'] == 42
    >>> assert GenericSettings.x == 42
    """

    default_settings = None

    @classmethod
    def process_value(cls, k, v):
        v = super().process_value(k, v)
        # NOTE we dont want to parse {} in strings for now
        # if isinstance(v, str):
        #     return v.format(**os.environ)
        return v

    @classmethod
    def from_yaml(
        cls, filename: str, spec: BaseModel = BlurringAsAServiceSettingsSpec
    ) -> "GenericSettings":
        """Read the config file and returns it as dictionary"""
        try:
            with open(filename, "r") as f:
                cfg = yaml.safe_load(f)
        except OSError as e:
            logger.critical(f"Could not open config file {filename}: {e}. Aborting.")
            raise
        except yaml.YAMLError as e:
            logger.critical(f"Error parsing config in {filename}: {e}. Aborting.")
            raise
        cfg = cls.validate(cfg, spec=spec)
        settings = cls(cfg)
        return settings

    @classmethod
    def validate(
        cls,
        data: Dict,
        errors: str = "raise",
        spec: BaseModel = BlurringAsAServiceSettingsSpec,
    ) -> Dict:
        """Called when setting the global settings. Can be overriden in subclasses.

        Might modify data in-place.

        Parameters
        ----------
        data: Dict
            the data to validate
        errors : {"log", "raise"}
            what to do with errors. Default: "raise"
        spec : pydantic.BaseModel, optional
            if provided, the data is validated using this model. Default:
            `BlurringAsAServiceSettingsSpec`.

        Raises
        ------
        - a ConfigurationError when any issues are encountered.

        Returns
        -------
        data : dict
            the validated data. Might be modified.
        """
        if spec:
            try:
                data = spec.validate(data).dict()
            except Exception as e:
                if errors == "raise":
                    raise ConfigurationError(
                        f"Encountered the following error when validating the settings: {e}."
                    )
                else:
                    logger.error(f"Error parsing the settings: {e}")
        return data

    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = BlurringAsAServiceSettingsSpec
    ) -> "GenericSettings":
        """Load settings from yaml and register them as the default"""
        settings = cls.from_yaml(filename, spec=spec)
        cls.set_settings(settings)
        return settings

    @staticmethod
    def set_settings(settings: AttrDict) -> None:
        if not isinstance(settings, AttrDict):
            settings = AttrDict(settings)
        GenericSettings.default_settings = settings

    @classmethod
    def get_settings(cls) -> Optional[AttrDict]:
        return cls.default_settings


class Settings(GenericSettings):  # type: ignore
    """Settings implementation specifically for dataset configurations.

    Supplies a custom validate method.
    """

    @classmethod
    def validate(
        cls,
        data: Dict,
        errors: str = "raise",
        spec: BaseModel = BlurringAsAServiceSettingsSpec,
    ) -> Dict:
        """A non-exhaustive validity check for the settings dictionary."""
        data = super().validate(data, errors, spec)
        logger.debug("Validating settings.")
        return data


# register an empty Settings object as default.
# Custom packages will most likely change this default...
Settings.set_settings(Settings())


def setup_logging(cfg: Dict[str, Any]):
    """Sets up logging according to the configuration.

    Parameters
    ----------
    cfg:
        configuration part of the config.yml

    Returns
    -------
    None

    """
    logging.basicConfig(**cfg["basic_config"])
    for pkg in cfg["own_packages"]:
        logging.getLogger(pkg).setLevel(cfg["loglevel_own"])
    for logger_, level in cfg["extra_loglevels"].items():
        logging.getLogger(logger_).setLevel(level)


def strings2flags(
    string_flags: List[str], flag_cls: Type[IntFlag], empty=0
) -> Type[IntFlag]:
    flags = [flag_cls[f] for f in string_flags]
    flags = reduce(lambda x, y: x | y, flags, empty)  # type: ignore
    return flags  # type: ignore
