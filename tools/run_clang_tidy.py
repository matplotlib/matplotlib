#!/usr/bin/env python3
"""
Run clang-tidy on Matplotlib's C/C++ and Objective-C sources.

Build directory
---------------
The script uses ``build/clang-tidy/`` as a dedicated meson build directory.
If it does not exist it is created automatically via ``meson setup``
(no compilation takes place).

Prerequisites
-------------
``meson``, ``pybind11``, and ``clang-tidy`` must be available.

Usage
-----
::

    python tools/run_clang_tidy.py
"""

import re
import subprocess
import sys
from pathlib import Path

_VENDORED_RE = re.compile(r"[\\/](extern|subprojects)[\\/]")
_DIAG_RE = re.compile(r"\S.*: (?:warning|error):")


def _filter_clang_tidy_output(text: str) -> str:
    """
    Filter clang-tidy output to drop entire diagnostic chains that originate
    in vendored code (``extern/`` or ``subprojects/``).

    clang-tidy output is structured as diagnostic groups: a ``warning:`` or
    ``error:`` line followed by ``note:`` lines and source snippets, until the
    next ``warning:``/``error:`` or end of output.  If the triggering
    diagnostic is in vendored code, the entire group (including notes that
    reference our own ``src/`` files as execution-path context) is dropped.
    """
    groups: list[list[str]] = []
    current: list[str] = []

    for line in text.splitlines():
        if _DIAG_RE.match(line):
            if current:
                groups.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        groups.append(current)

    kept: list[str] = []
    for group in groups:
        header = group[0]
        if _DIAG_RE.match(header) and _VENDORED_RE.search(header):
            continue
        kept.extend(group)

    return "\n".join(kept)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = repo_root / "build" / "clang-tidy"

    result = subprocess.run(
        ["meson", "setup", "--reconfigure", str(build_dir)],
        cwd=repo_root,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)

    result = subprocess.run(
        [
            "meson",
            "--internal",
            "clangtidy",
            str(repo_root),
            str(build_dir),
            "--color",
            "never",
        ],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )

    filtered = _filter_clang_tidy_output(result.stdout)
    if filtered:
        print(filtered)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
