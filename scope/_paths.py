"""Helpers for resolving packaged runtime data paths."""

from pathlib import Path


def get_package_root() -> Path:
    """Return the installed ``scope`` package directory."""
    return Path(__file__).resolve().parent


def get_default_input_dir() -> Path:
    """Resolve the default SCOPE input directory.

    Preference order:
    1. Packaged runtime data inside ``scope/input``
    2. Source checkout data at repository root ``input/``
    """
    package_input = get_package_root() / "input"
    if package_input.exists():
        return package_input

    repo_input = get_package_root().parent / "input"
    if repo_input.exists():
        return repo_input

    return package_input
