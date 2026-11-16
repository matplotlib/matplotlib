#!/usr/bin/env python3
"""
Tool to install macOS packages on GitHub runners

Usage: python tools/install_macos_packages.py <command>

Commands:
install-cache-only
install-all
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Homebrew packages that will be installed on every run. By default, the dependencies
# of these packages will be included in the GitHub Actions cache.
PACKAGES = (
    "ccache", "ffmpeg", "ghostscript", "gpg",
    "gobject-introspection", "gtk4",
    "imagemagick", "ninja"
)

# Additional packages to include in the cache
CACHE_INCLUDE = ("gpg",)

# Packages to explicitly exclude from the cache due to their size.
CACHE_EXCLUDE = ("python", "icu4c", "openssl")

# Homebrew cask names of fonts to install
FONT_PACKAGES = ("font-noto-sans-cjk", "font-noto-sans-cjk-sc")

# Homebrew cask name of Inkscape
INKSCAPE_PACKAGE = "inkscape"


def run_brew_info(*args: str) -> dict:
    result = subprocess.run(
        ["brew", "info", "--json=v2", *args],
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(result.stdout)


class PackageFilter:
    _packages: list[str]
    _excludes: set[str]
    _dependencies: dict[str, list[str]]
    _name_map: dict[str, str]
    _tainted_cache: dict[str, bool]

    def __init__(self, packages: Sequence[str], excludes: Sequence[str]) -> None:
        self._packages = list(packages)
        self._dependencies = {}
        self._name_map = {}
        self._tainted_cache = {}

        self._fill_graph()

        self._excludes = {self._resolve_name(e) for e in excludes}

    def _fill_graph(self) -> None:
        requested_packages = set(self._packages)
        pending_packages = list(self._packages)

        while pending_packages:
            info = run_brew_info("--formula", *pending_packages)

            for formula in info["formulae"]:
                dependencies = formula.get("dependencies", [])
                full_name = formula["full_name"]
                self._name_map[full_name] = full_name
                self._name_map[formula["name"]] = full_name
                for alias in formula["aliases"]:
                    self._name_map[alias] = full_name
                self._dependencies[full_name] = dependencies

            pending_packages = sorted({
                dep for deps in self._dependencies.values() for dep in deps
                if dep not in requested_packages
            })

            requested_packages.update(pending_packages)

    def _resolve_name(self, name: str) -> str:
        name = name.split("@")[0]
        return self._name_map.get(name, name)

    def is_excluded(self, name: str) -> bool:
        return self._resolve_name(name) in self._excludes

    def is_tainted(self, name: str) -> bool:
        cache = self._tainted_cache
        name = self._resolve_name(name)

        if name in cache:
            return cache[name]

        # Mark the formula as untainted before recursing to prevent an infinite loop
        cache[name] = False
        cache[name] = any(
            self.is_excluded(dep) or self.is_tainted(dep)
            for dep in self._dependencies.get(name, [])
        )

        return cache[name]

    def get_filtered_packages(self) -> list[str]:
        result = set()
        seen = set()
        stack = list(self._packages)

        while stack:
            name = self._resolve_name(stack.pop())
            if name in seen:
                continue
            seen.add(name)
            for dep in self._dependencies.get(name, []):
                if dep in result:
                    continue
                if not self.is_excluded(dep) and not self.is_tainted(dep):
                    result.add(dep)
                stack.append(dep)

        return sorted(result)


def download_cask(package_name: str, dest: Path) -> None:
    info = run_brew_info("--cask", package_name)
    cask = info["casks"][0]
    url = cask["url"]
    expected_hash = cask["sha256"]

    urllib.request.urlretrieve(url, dest)
    actual_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
    if (expected_hash != "no_check") and (actual_hash != expected_hash):
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {expected_hash}, got {actual_hash}"
        )


class Task:
    name: str
    out: list[str]
    err: list[str]

    def __init__(self, name: str) -> None:
        self.name = name
        self.out = []
        self.err = []
        self._start = time.perf_counter()

    def execute(self, tmp_path: Path) -> None:
        raise NotImplementedError

    def run(self, *args: str) -> None:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        def pump(stream, sink: list[str]) -> None:
            for line in stream:
                sink.append(f"[{time.perf_counter() - self._start:8.2f}s] {line}")

        threads = [
            threading.Thread(target=pump, args=(proc.stdout, self.out)),
            threading.Thread(target=pump, args=(proc.stderr, self.err)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, args)

    def __call__(self) -> tuple[int, str, str, float]:
        try:
            with tempfile.TemporaryDirectory(delete=False) as tmp:
                tmp_path = Path(tmp)
                self.execute(tmp_path)
            code = 0
        except subprocess.CalledProcessError as e:
            code = e.returncode
        except Exception as e:
            self.err.append(f"{e}\n")
            code = 1
        return (
            code,
            "".join(self.out),
            "".join(self.err),
            time.perf_counter() - self._start,
        )


class InstallPackagesTask(Task):

    def __init__(self, name: str, packages: Sequence[str]) -> None:
        super().__init__(name)
        self._packages = list(packages)

    def execute(self, tmp_path: Path) -> None:
        self.run("brew", "install", "--display-times", *self._packages)


class InstallFontTask(Task):

    def __init__(self, package_name: str) -> None:
        super().__init__(package_name)
        self._package_name = package_name

    def execute(self, tmp_path: Path) -> None:
        archive = tmp_path / "font.zip"
        download_cask(self._package_name, archive)

        extracted = tmp_path / "extracted"
        self.run("unzip", "-o", str(archive), "-d", str(extracted))

        font_dir = Path.home() / "Library" / "Fonts"
        font_dir.mkdir(parents=True, exist_ok=True)
        for path in extracted.rglob("*"):
            if path.suffix.lower() in {".ttc", ".ttf", ".otf"}:
                shutil.copy2(path, font_dir / path.name)
                self.out.append(f"Installed {path.name}\n")


class InstallInkscapeTask(Task):

    def __init__(self, package_name: str) -> None:
        super().__init__(package_name)
        self._package_name = package_name

    def execute(self, tmp_path: Path) -> None:
        dmg = tmp_path / "Inkscape.dmg"
        mnt = tmp_path / "mnt"
        mnt.mkdir()

        download_cask(self._package_name, dmg)

        self.run("hdiutil", "attach", "-readonly",
                 "-nobrowse", "-noverify", "-noautofsck",
                 "-mountpoint", str(mnt), str(dmg))
        self.run("ditto", str(mnt / "Inkscape.app"), "/Applications/Inkscape.app")


def install_cache_only_packages():
    package_filter = PackageFilter(PACKAGES, CACHE_EXCLUDE)
    filtered_packages = package_filter.get_filtered_packages()
    filtered_packages = sorted(set(filtered_packages) | set(CACHE_INCLUDE))

    code, out, err, _ = InstallPackagesTask("cache-only packages", filtered_packages)()
    print(out, end="")
    print(err, end="")
    if code:
        sys.exit(code)


def install_all_packages():
    tasks = [
        InstallPackagesTask("all packages", PACKAGES),
        *(InstallFontTask(package) for package in FONT_PACKAGES),
        InstallInkscapeTask(INKSCAPE_PACKAGE),
    ]
    exit_code = 0
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [(task.name, executor.submit(task)) for task in tasks]
        for name, future in futures:
            code, out, err, elapsed = future.result()
            print(f"::group::{name} (exit {code}, {elapsed:.2f}s)")
            if out:
                print(out, end="")
            if err:
                print(err, end="")
            print("::endgroup::")
            sys.stdout.flush()
            exit_code = exit_code or code
    if exit_code:
        sys.exit(exit_code)


def main():
    if len(sys.argv) < 2:
        exit("No command provided")

    command = sys.argv[1]
    if command == "install-cache-only":
        install_cache_only_packages()

    elif command == "install-all":
        install_all_packages()

    else:
        exit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
