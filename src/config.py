"""Load a YAML run config and apply command-line overrides."""

from pathlib import Path

import yaml


def apply_override(config: dict, override: str) -> None:
    """Set one dotted field in place, e.g. ``loop.max_iterations=2``.

    Args:
        config: nested config dict, modified in place.
        override: ``section.key=value``; the value is parsed as YAML.

    Returns:
        None.
    """
    dotted_key, separator, raw_value = override.partition("=")
    if not separator:
        raise ValueError(f"override must look like section.key=value, got {override!r}")
    *parents, leaf = dotted_key.split(".")
    node = config
    for parent in parents:
        if parent not in node:
            raise KeyError(f"unknown config section {parent!r} in {override!r}")
        node = node[parent]
    if leaf not in node:
        raise KeyError(f"unknown config field {leaf!r} in {override!r}")
    node[leaf] = yaml.safe_load(raw_value)


def load_config(path: Path, overrides: list[str]) -> dict:
    """Read a config file and apply overrides in order.

    Args:
        path: YAML config file.
        overrides: list of ``section.key=value`` strings.

    Returns:
        The resulting config dict.
    """
    config = yaml.safe_load(path.read_text())
    for override in overrides:
        apply_override(config, override)
    return config
