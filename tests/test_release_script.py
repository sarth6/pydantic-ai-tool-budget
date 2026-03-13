from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "release.py"
SPEC = importlib.util.spec_from_file_location("release_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
release_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_script
SPEC.loader.exec_module(release_script)


def test_parse_valid_version() -> None:
    version = release_script.Version.parse("1.2.3")
    assert str(version) == "1.2.3"


@pytest.mark.parametrize("raw", ["1.2", "1.2.3.4", "v1.2.3", "01.2.3"])
def test_parse_invalid_version(raw: str) -> None:
    with pytest.raises(release_script.ReleaseError):
        release_script.Version.parse(raw)


@pytest.mark.parametrize(
    ("bump", "expected"),
    [("patch", "0.1.1"), ("minor", "0.2.0"), ("major", "1.0.0")],
)
def test_resolve_next_version_from_bump(bump: str, expected: str) -> None:
    current = release_script.Version.parse("0.1.0")
    next_version = release_script.resolve_next_version(current, bump, None)
    assert str(next_version) == expected


def test_resolve_custom_version() -> None:
    current = release_script.Version.parse("0.1.0")
    next_version = release_script.resolve_next_version(current, "custom", "0.3.0")
    assert str(next_version) == "0.3.0"


def test_resolve_custom_requires_version() -> None:
    current = release_script.Version.parse("0.1.0")
    with pytest.raises(release_script.ReleaseError):
        release_script.resolve_next_version(current, "custom", None)


def test_write_version_updates_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\nversion = "0.1.0"\n')

    release_script.write_version(pyproject, release_script.Version.parse("0.1.1"))

    assert 'version = "0.1.1"' in pyproject.read_text()
