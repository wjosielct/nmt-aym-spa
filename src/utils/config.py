# =====================
# ./src/utils/config.py
# =====================

from pathlib import Path
from typing import Any

import yaml


def load_config(
        config_path: str | Path,
) -> dict[str, Any]:
    """
    Loads a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    ValueError
        If the path is not a file, the YAML is empty,
        or the YAML syntax is invalid.

    TypeError
        If the root YAML object is not a dictionary/mapping.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Configuration path is not a file: {config_path}"
        )

    try:
        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:

            config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML syntax in configuration file: "
            f"{config_path}"
        ) from exc

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    if not isinstance(config, dict):
        raise TypeError(
            "The root of the YAML configuration must be "
            f"a mapping/dictionary: {config_path}"
        )

    return config