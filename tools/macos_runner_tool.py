#!/usr/bin/env python3
"""
Optimize the performance of GitHub macOS runners.

Run as ``python tools/macos_runner_tool.py <command>``.

Notes
-----
The following commands are available.

disable-services
    Disables various unneeded macOS system services which waste
    resources better spent installing packages or compiling.

install-cache-only-packages
    Prepares our GitHub Actions cache by installing all dependencies
    of PACKAGES except for those which would trigger the installation
    of those in CACHE_EXCLUDE. Packages in CACHE_INCLUDE are also installed.

    This command is intended to be used by the macos-cache.yml Action.

install-all-packages
    Performs the following in parallel:

    - Uses `brew install` to install all PACKAGES.
    - Manually installs each font specified by FONT_CASKS.
    - Manually installs Inkscape.

    This command is intended to be used by test runners.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Homebrew packages that will be installed on every run. By default, the dependencies
# of these packages will be included in the GitHub Actions cache.
PACKAGES = (
    "ccache", "ffmpeg", "ghostscript",
    "gobject-introspection", "gtk4",
    "imagemagick", "ninja"
)

# Additional packages to include in the cache
CACHE_INCLUDE = ("gobject-introspection",)

# Packages to explicitly exclude from the cache due to their size.
CACHE_EXCLUDE = ("icu4c", "openssl")

# Homebrew cask names of fonts to install
FONT_CASKS = ("font-noto-sans-cjk", "font-noto-sans-cjk-sc")

# Homebrew cask name of Inkscape
INKSCAPE_CASK = "inkscape"

# System services to disable
SERVICES = (
    # Safari Bookmark Sync is chewing CPU while we are building matplotlib.
    ("gui", "com.apple.Safari.History"),
    ("gui", "com.apple.Safari.PasswordBreachAgent"),
    ("gui", "com.apple.Safari.SafeBrowsing.Service"),
    ("gui", "com.apple.SafariBookmarksSyncAgent"),
    ("gui", "com.apple.SafariLaunchAgent"),

    # These were also chewing CPU
    ("gui", "com.apple.helpd"),  # Help Daemon / help documentation
    ("gui", "com.apple.homed"),  # HomeKit / smart home
    ("gui", "com.apple.weatherd"),  # Weather widget
    ("gui", "com.apple.triald"),  # A/B testing, telemetry
    ("system", "com.apple.triald.system"),

    # Image file analysis / VisualIntelligence
    ("gui", "com.apple.mediaanalysisd"),
    ("gui", "com.apple.visualintelligenced"),

    # Siri Search / Knowledge Graph / Apple Intelligence / etc
    ("gui", "com.apple.duetexpertd"),
    ("gui", "com.apple.intelligencecontextd"),
    ("gui", "com.apple.intelligenceflowd"),
    ("gui", "com.apple.intelligenceplatformd"),
    ("gui", "com.apple.intelligencetasksd"),
    ("gui", "com.apple.knowledge-agent"),
    ("gui", "com.apple.knowledgeconstructiond"),
    ("gui", "com.apple.parsec-fbf"),
    ("gui", "com.apple.parsecd"),
    ("gui", "com.apple.proactived"),
    ("gui", "com.apple.proactiveeventtrackerd"),
    ("gui", "com.apple.siriactionsd"),
    ("gui", "com.apple.siriappintentsd"),
    ("gui", "com.apple.siriinferenced"),
    ("gui", "com.apple.siriknowledged"),
    ("gui", "com.apple.sirittsd"),
    ("gui", "com.apple.spotlightknowledged"),
    ("gui", "com.apple.suggestd"),
    ("system", "com.apple.coreduetd"),
    ("user", "com.apple.siri.context.service"),
    ("user", "com.apple.siri-distributed-evaluation"),
    ("user", "com.apple.routined"),
)

# Lingering processes to kill after disabling system services.
# Exercise caution as `killall -9 -m` uses partial matching.
PROCESSES_TO_KILL = (
    r"^[Ss]iri.*",
    r"Safari",
    r"Intelligence",
)


def run_brew_info(*args: str) -> dict:
    """Return Homebrew package metadata for the requested formulae or casks."""
    result = subprocess.run(
        ["brew", "info", "--json=v2", *args],
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(result.stdout)


class PackageFilter:
    """
    Generate a filtered list of Homebrew dependencies.

    The list contains all dependencies of the specified base `packages`,
    minus the `excludes` and everything that depends on them.

    Consider the following dependency graph:

    A       B   C
    |           |
    +---+       +---+
    |   |       |   |
    D   E       F   G
                    |
                    +---+
                    |   |
                    H   I
                    |
                    J

    (A, B, C) will never be included in the result as they are base packages.

    If we exclude H: both H (the excluded package) and G (its dependent)
    will be excluded. The end result will be: (D, E, F, I, J)

    The goal of this class is to install as many Homebrew packages as possible
    in our cache without triggering the inclusion of large packages.

    Parameters
    ----------
    packages : sequence of str
        The base packages whose dependencies are considered.
    excludes : sequence of str
        The packages to exclude, along with any package that depends on them.
    """

    _packages: list[str]
    _excludes: set[str]
    _dependencies: dict[str, list[str]]
    _name_map: dict[str, str]

    def __init__(self, packages: Sequence[str], excludes: Sequence[str]) -> None:
        self._packages = list(packages)
        self._dependencies = {}
        self._name_map = {}

        self._fill_graph()

        self._excludes = {self._resolve_name(e) for e in excludes}

    def _fill_graph(self) -> None:
        """Populate ``_name_map`` and ``_dependencies`` using `brew info`."""
        requested_packages = set(self._packages)
        pending_packages = list(self._packages)

        # As an optimization, run `brew info` on one "layer" at a time
        # rather than per-formula.
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
        """Return the canonical full_name of the named package."""
        name = name.split("@")[0]
        return self._name_map.get(name, name)

    def is_excluded(self, name: str) -> bool:
        """Return whether the named package is in the exclude list."""
        return self._resolve_name(name) in self._excludes

    def has_excluded_dependency(self, name: str) -> bool:
        """Return whether the named package has a dependency in the exclude list."""
        name = self._resolve_name(name)
        seen = {name}
        stack = [name]

        while stack:
            for dep in self._dependencies.get(stack.pop(), []):
                if self.is_excluded(dep):
                    return True
                dep = self._resolve_name(dep)
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)

        return False

    def get_filtered_packages(self) -> list[str]:
        """
        Return all packages installable without installing an excluded one.

        Returns
        -------
        list of str
            The sorted names of the packages that can be installed without
            triggering the installation of an excluded package.
        """
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
                if not self.is_excluded(dep) and not self.has_excluded_dependency(dep):
                    result.add(dep)
                stack.append(dep)

        return sorted(result)


def download_cask(name: str, destination: Path) -> None:
    """
    Download the named Homebrew cask to the specified destination.

    The URL and SHA-256 hash of the cask are looked up with `brew info`.
    """
    ATTEMPTS = 3
    TIMEOUT = 60

    info = run_brew_info("--cask", name)
    cask = info["casks"][0]
    url = cask["url"]
    expected_hash = cask["sha256"]

    for attempt in range(ATTEMPTS):
        try:
            timeout = TIMEOUT
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with destination.open("wb") as f:
                    shutil.copyfileobj(response, f)
            break
        except OSError as e:
            permanent = isinstance(e, urllib.error.HTTPError) and e.code < 500
            if permanent or attempt == ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)

    actual_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    if (expected_hash != "no_check") and (actual_hash != expected_hash):
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {expected_hash}, got {actual_hash}"
        )


def run_tasks(*tasks: Task) -> None:
    """
    Run the specified tasks in parallel using a ThreadPoolExecutor.

    The output of each task is printed using GitHub Actions grouping syntax.
    """
    exit_code = 0
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [(task.name, executor.submit(task)) for task in tasks]
        for name, future in futures:
            code, out, elapsed = future.result()
            print(f"::group::{name} (exit {code}, {elapsed:.2f}s)")
            if out:
                print(out, end="")
            print("::endgroup::")
            sys.stdout.flush()
            exit_code = exit_code or code
    if exit_code:
        sys.exit(exit_code)


class Task:
    """Represent an abstract task. Used with `run_tasks`."""
    name: str
    out: list[str]

    def __init__(self, name: str) -> None:
        self.name = name
        self.out = []
        self._start = time.perf_counter()

    def execute(self, tmp_path: Path) -> None:
        """
        Perform the work of the task.

        Parameters
        ----------
        tmp_path : pathlib.Path
            A temporary directory available to the task.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def run(self, *args: str) -> None:
        """
        Run a subprocess and collect its output with timestamps.

        Parameters
        ----------
        *args : str
            The command and its arguments.

        Raises
        ------
        subprocess.CalledProcessError
            If the subprocess exits with a non-zero status.
        """
        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as proc:
            for line in proc.stdout:
                self.out.append(f"[{time.perf_counter() - self._start:8.2f}s] {line}")
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, args)

    def run_string(self, cmd: str, *args: str) -> None:
        """Pass the specified string into `bash -c`"""
        self.run("bash", "-c", cmd, "bash", *args)

    def run_ignoring_errors(self, *args: str) -> None:
        """Run a subprocess, ignoring a non-zero exit status."""
        try:
            self.run(*args)
        except subprocess.CalledProcessError:
            pass

    def __call__(self) -> tuple[int, str, float]:
        try:
            with tempfile.TemporaryDirectory(delete=False) as tmp:
                tmp_path = Path(tmp)
                self.execute(tmp_path)
            code = 0
        except subprocess.CalledProcessError as e:
            code = e.returncode
        except Exception as e:
            self.out.append(f"{e}\n")
            code = 1
        return (code, "".join(self.out), time.perf_counter() - self._start)


class CleanRunnerTask(Task):
    """Clean pre-installed Homebrew packages from GitHub runner."""

    def __init__(self) -> None:
        super().__init__("clean runner")

    def execute(self, tmp_path: Path) -> None:
        # Delete the contents of various homebrew subdirectories.
        # `mv ... /opt/homebrew-aside` is used for performance (fs rename)
        REMOVE_BREW_SUBDIR = r"""
            set -e
            sudo mkdir -p "/opt/homebrew-aside/$1"

            if [ -d "/opt/homebrew/$1" ]; then
                sudo find "/opt/homebrew/$1" -mindepth 1 -maxdepth 1 \
                    -exec sh -c 'mv "$@" "$0"' "/opt/homebrew-aside/$1/" {} +
            fi
        """
        for subdir in (
            "Caskroom", "Cellar", "include", "opt", "var/homebrew/linked", "test"
        ):
            self.run_string(REMOVE_BREW_SUBDIR, subdir)

        # Clear all symlinks
        self.run_string(r"""
            find /opt/homebrew \( -path /opt/homebrew/completions \
                -o -path /opt/homebrew/Library \
                -o -path /opt/homebrew/Frameworks \
                -o -path /opt/homebrew/share \) -prune \
                -o -type l -exec rm {} \;
        """)


class PreserveToolsTask(Task):
    """
    Preserve select tools by copying to /usr/local and rewriting linker entries.
    This is necessary as GitHub's actions/cache/restore needs both gtar and unzstd
    """

    def __init__(self) -> None:
        super().__init__("preserve tools")

    def execute(self, tmp_path: Path) -> None:
        HOMEBREW_PATH = Path("/opt/homebrew")
        OUT_BIN_DIR = Path("/usr/local/bin")
        OUT_LIB_DIR = Path("/usr/local/lib")
        BINS = ("zstd", "zstdmt", "unzstd", "gtar")
        LIBS = ("libzstd.1.dylib", "liblzma.5.dylib", "liblz4.1.dylib")

        def get_linker_entries_to_rewrite(path: Path):
            cmd = ("otool", "-L", str(path))
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            entries = [
                line.split(" (compatibility", 1)[0].strip()
                for line in result.stdout.splitlines()[1:]
            ]
            return [e for e in entries if e.strip().startswith(str(HOMEBREW_PATH))]

        def rewrite_linker_entries(path: Path):
            to_rewrite = get_linker_entries_to_rewrite(path)

            if len(to_rewrite) == 0:
                return

            args: list[str | Path] = []
            for entry in to_rewrite:
                args += ["-change", entry, OUT_LIB_DIR / Path(entry).name]

            self.run("sudo", "install_name_tool", *args, path)
            self.run("sudo", "codesign", "--force", "--sign", "-", path)

        self.run("sudo", "mkdir", "-p", OUT_BIN_DIR, OUT_LIB_DIR)

        for bin in BINS:
            self.run("sudo", "cp", "-L", HOMEBREW_PATH / "bin" / bin, OUT_BIN_DIR)
            self.run("sudo", "chmod", "u+w", OUT_BIN_DIR / bin)

        for lib in LIBS:
            self.run("sudo", "cp", "-L", HOMEBREW_PATH / "lib" / lib, OUT_LIB_DIR)
            self.run("sudo", "chmod", "u+w", OUT_LIB_DIR / lib)

        for bin in BINS:
            rewrite_linker_entries(OUT_BIN_DIR / bin)

        for bin in BINS:
            self.run("file", OUT_BIN_DIR / bin)
            self.run(OUT_BIN_DIR / bin, "--version")
            self.run("otool", "-L", OUT_BIN_DIR / bin)


class DisableServicesTask(Task):
    """Disable various unneeded macOS system services."""

    def __init__(self, services: Sequence[tuple[str, str]]) -> None:
        super().__init__("disable services")
        self._services = list(services)

    def execute(self, tmp_path: Path) -> None:
        for domain, name in self._services:
            if domain not in {"gui", "user", "system"}:
                raise ValueError(f"Unsupported domain: {domain}")

            uid = "" if domain == "system" else f"{os.getuid()}/"
            specifier = f"{domain}/{uid}{name}"

            self.run_ignoring_errors("sudo", "launchctl", "bootout", specifier)
            self.run_ignoring_errors("sudo", "launchctl", "disable", specifier)

        for pattern in PROCESSES_TO_KILL:
            self.run_ignoring_errors("sudo", "killall", "-9", "-m", pattern)


class InstallPackagesTask(Task):
    """Perform a `brew install` of the specified packages."""

    def __init__(self, name: str, packages: Sequence[str]) -> None:
        super().__init__(name)
        self._packages = list(packages)

    def execute(self, tmp_path: Path) -> None:
        self.run("brew", "install", "--display-times", *self._packages)


class InstallFontTask(Task):
    """Download and install a font."""

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
    """Download and install Inkscape."""

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
        self.run("ln", "-s",
                 "/Applications/Inkscape.app/Contents/MacOS/inkscape",
                 "/usr/local/bin")


def main() -> None:
    if len(sys.argv) < 2:
        exit("No command(s) provided")

    for command in sys.argv[1:]:
        if command == "clean-runner":
            run_tasks(CleanRunnerTask())

        elif command == "preserve-tools":
            run_tasks(PreserveToolsTask())

        elif command == "disable-services":
            run_tasks(DisableServicesTask(SERVICES))

        elif command == "install-cache-only-packages":
            package_filter = PackageFilter(PACKAGES, CACHE_EXCLUDE)
            filtered_packages = package_filter.get_filtered_packages()
            filtered_packages = sorted(set(filtered_packages) | set(CACHE_INCLUDE))
            run_tasks(InstallPackagesTask("cache-only packages", filtered_packages))

        elif command == "install-all-packages":
            run_tasks(
                InstallPackagesTask("all packages", PACKAGES),
                *(InstallFontTask(cask) for cask in FONT_CASKS),
                InstallInkscapeTask(INKSCAPE_CASK),
            )

        else:
            sys.exit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
