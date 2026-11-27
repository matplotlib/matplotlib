#!/usr/bin/env python3
"""
Run clang-include-cleaner on Matplotlib's C++ sources and headers.

Build directory
---------------
The script uses ``build/check-cpp-includes/`` as a dedicated meson build directory.
If it does not exist it is created automatically via ``meson setup``

Prerequisites
-------------
``meson`` and ``clang-include-cleaner`` must be available.

Notes
-----
This tool only checks standard C++ headers and user headers starting with "agg_".
Subprojects (pybind11, libraqm, freetype, etc) and platform-specific headers
are filtered out and should be manually maintained.

Usage
-----
::

    python tools/check_cpp_includes.py
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

enable_color = True

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DEFAULT_BUILD_DIR = ROOT_DIR / "build" / "check-cpp-includes"

EXCLUDES = [
    # Ignore all user headers except for "agg_*"
    r'^"(?!agg_)',

    # Ignore various subprojects
    r'^["<]pybind11\/',
    r'^["<]freetype\/',
    r'^<ft2build\.h>',
    r'^<raqm.h>',

    # Ignore <Python.h> and platform headers like <CoreFoundation/*>
    r'^<[A-Z]'
]

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"][^>"]+[>"])', re.MULTILINE)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DARK_GREY = "\033[90m"
RESET = "\033[0m"
WHITE_ON_BLUE = "\033[37;44m"


@dataclass
class SourceFile:
    path: Path
    original: set[str] = field(default_factory=set)
    all: set[str] = field(default_factory=set)
    additions: set[str] = field(default_factory=set)
    removals: set[str] = field(default_factory=set)
    error_msg: str | None = None

    def has_changes(self) -> bool:
        return len(self.additions) > 0 or len(self.removals) > 0


def find_tool(name: str, required: bool) -> str:
    path = shutil.which(name)
    if not path and required:
        sys.exit(f"{name} not found in PATH")
    return path


def configure_meson(build_dir):
    cmd_args = [find_tool("meson", required=True), "setup"]
    if (build_dir / "meson-private").exists():
        cmd_args.append("--reconfigure")
    cmd_args.append(str(build_dir))

    env = os.environ.copy()
    if lld := find_tool("lld", required=False):
        for var in ("CC_LD", "CXX_LD", "OBJC_LD"):
            env[var] = "lld"

    result = subprocess.run(cmd_args, cwd=ROOT_DIR, env=env,
                            capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        sys.exit("meson setup failed")


def process_source_file(
    source_file: SourceFile,
    clang_include_cleaner: Path,
    build_dir: Path
) -> None:
    cmd_args = (
        clang_include_cleaner,
        "--print=changes",
        "-p", str(build_dir),
        str(source_file.path.resolve())
    )

    result = subprocess.run(cmd_args, capture_output=True, text=True)

    if result.returncode != 0:
        source_file.error_msg = result.stderr
        return

    pattern = re.compile(r'^([+-]) ([<"][^>"]+[>"])')
    for line in result.stdout.splitlines():
        if match := pattern.match(line):
            if match[1] == "+":
                source_file.additions.add(match[2])
            elif match[1] == "-":
                source_file.removals.add(match[2])

    try:
        source_file.original = set(INCLUDE_RE.findall(source_file.path.read_text()))
    except OSError:
        pass


def apply_excludes(source_file: SourceFile, excludes: list[re.Pattern[str]]) -> None:
    for s in source_file.original, source_file.additions, source_file.removals:
        s.difference_update(
            entry for entry in s.copy()
            if any(exclude.search(entry) for exclude in excludes)
        )


def colorize(text: str, color: str) -> str:
    if enable_color:
        return f"{color}{text}{RESET}"
    else:
        return text


def print_updated_includes(source_file: SourceFile) -> None:
    includes = (source_file.original - source_file.removals) | source_file.additions

    groups = [
        sorted(inc for inc in includes if inc.startswith("<")),
        sorted(inc for inc in includes if inc.startswith('"')),
    ]

    for index, group in enumerate(g for g in groups if g):
        if index:
            print()
        for include in group:
            line = f"#include {include}"
            if include in source_file.additions:
                line = colorize(line, GREEN)
            print(line)


def report_unchanged_source_files(source_files: list[SourceFile]) -> None:
    print("")
    print("The following files are ok:")
    for source_file in source_files:
        print(source_file.path.relative_to(ROOT_DIR))
    print("")


def report_source_file(source_file: SourceFile) -> None:
    print(colorize("-" * 80, DARK_GREY))
    print()
    print(colorize(source_file.path.relative_to(ROOT_DIR), WHITE_ON_BLUE))
    print()

    has_additions = len(source_file.additions) > 0
    has_removals = len(source_file.removals) > 0

    if has_removals > 0:
        print(colorize("Removals:", YELLOW))
        for include in source_file.removals:
            print(colorize(f"#include {include}", RED))
        print()

    if has_additions > 0:
        print(colorize("Additions:", YELLOW))
        for include in source_file.additions:
            print(colorize(f"#include {include}", GREEN))
        print()

    if has_additions or has_removals:
        print(colorize("Updated includes:", YELLOW))
        print_updated_includes(source_file)
        print()


def main():
    global enable_color

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--no-color", action="store_true", default=False)
    args = parser.parse_args()

    enable_color = not args.no_color

    clang_include_cleaner = find_tool("clang-include-cleaner", required=True)
    build_dir = args.build_dir.resolve()

    print("Configuring meson...")
    configure_meson(build_dir)

    all_paths = sorted(SRC_DIR.rglob("*.h")) + sorted(SRC_DIR.rglob("*.cpp"))
    source_files = {}
    for path in all_paths:
        source_files[path] = SourceFile(path.resolve())

    print("Running clang-include-cleaner...")
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_source_file,
                source_file,
                clang_include_cleaner,
                build_dir
            )
            for _, source_file in source_files.items()
        ]

        for future in futures:
            future.result()

    # Filter excludes
    excludes_patterns = [re.compile(pattern) for pattern in EXCLUDES]
    for _, source_file in source_files.items():
        apply_excludes(source_file, excludes_patterns)

    # Check for errors
    for _, source_file in source_files.items():
        if source_file.error_msg:
            sys.exit(source_file.error_msg)

    unchanged_source_files = []
    changed_source_files = []

    for s in source_files.values():
        (changed_source_files if s.has_changes() else unchanged_source_files).append(s)

    if len(unchanged_source_files):
        unchanged_source_files.sort(key=lambda s: s.path)
        report_unchanged_source_files(unchanged_source_files)

    if len(changed_source_files):
        changed_source_files.sort(key=lambda s: s.path)
        for source_file in changed_source_files:
            report_source_file(source_file)


if __name__ == "__main__":
    main()
