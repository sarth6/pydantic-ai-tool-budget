from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
VERSION_LINE_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")$', re.MULTILINE)


class ReleaseError(ValueError):
    """Raised when release version input is invalid."""


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> Version:
        match = SEMVER_RE.fullmatch(raw)
        if match is None:
            raise ReleaseError(f"Invalid version: {raw!r}. Expected MAJOR.MINOR.PATCH.")
        major, minor, patch = (int(part) for part in match.groups())
        return cls(major=major, minor=minor, patch=patch)

    def bump(self, kind: str) -> Version:
        if kind == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        if kind == "minor":
            return Version(self.major, self.minor + 1, 0)
        if kind == "major":
            return Version(self.major + 1, 0, 0)
        raise ReleaseError(f"Unsupported bump kind: {kind!r}.")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def read_current_version(pyproject_path: Path) -> Version:
    for line in pyproject_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("version = "):
            _, value = stripped.split("=", maxsplit=1)
            return Version.parse(value.strip().strip('"'))
    raise ReleaseError(f"Could not find project version in {pyproject_path}.")


def resolve_next_version(current: Version, bump: str, custom_version: str | None) -> Version:
    if bump == "custom":
        if custom_version is None:
            raise ReleaseError("--version is required when bump is 'custom'.")
        next_version = Version.parse(custom_version)
    else:
        if custom_version is not None:
            raise ReleaseError("--version is only valid when bump is 'custom'.")
        next_version = current.bump(bump)

    if next_version == current:
        raise ReleaseError(f"Next version must differ from current version {current}.")
    return next_version


def write_version(pyproject_path: Path, new_version: Version) -> None:
    original = pyproject_path.read_text()
    updated, count = VERSION_LINE_RE.subn(rf"\g<1>{new_version}\g<3>", original, count=1)
    if count != 1:
        raise ReleaseError(f"Expected to update exactly one version line in {pyproject_path}, updated {count}.")
    pyproject_path.write_text(updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bump or set the package version in pyproject.toml.")
    parser.add_argument("bump", choices=("patch", "minor", "major", "custom"))
    parser.add_argument("--version", help="Explicit version to use when bump='custom'.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the computed version back to pyproject.toml. Without this flag, only print the next version.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to the pyproject.toml file to update.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pyproject_path = Path(args.pyproject)

    try:
        current = read_current_version(pyproject_path)
        next_version = resolve_next_version(current, args.bump, args.version)
        if args.write:
            write_version(pyproject_path, next_version)
    except ReleaseError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(next_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
