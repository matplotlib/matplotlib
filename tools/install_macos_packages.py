# Usage: ???
#
#

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CACHE_EXCLUDE = ("python")  # , "icu4c", "openssl", "openjpeg", "x265")

# URLs and SHA-256 hashes of fonts to install
FONTS = (
    {
        "url": "https://github.com/notofonts/noto-cjk/releases/download/"
               "Sans2.004/00_NotoSansCJK.ttc.zip",
        "hash": "a56077736038875f3a5b1e6715b2713a2bd049cc0bf0dfdf645927b070d38bfc"
    },
    {
        "url": "https://github.com/notofonts/noto-cjk/releases/download/"
               "Sans2.004/08_NotoSansCJKsc.zip",
        "hash": "a927e56f53bd6c3b920bc139c0b94aa36c7d9ad0cf009b159437a1a003581140"
    }
)

# The URL and SHA-256 hash of the Inkscape DMG
INKSCAPE = {
    "url": "https://media.inkscape.org/dl/resources/file/Inkscape-1.4.4_arm64.dmg",
    "hash": "eacca94ab01e59467cdd5452a2867c3734450f9a4cfb3346956cc9b07f34f3f1"
}


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

        self._get_brew_info()

        self._excludes = {self._resolve_name(e) for e in excludes}

    def _get_brew_info(self) -> None:
        requested_packages = set(self._packages)
        pending_packages = list(self._packages)

        while pending_packages:
            brew_info_result = subprocess.run(
                ["brew", "info", "--json=v2", "--formula", *pending_packages],
                check=True,
                capture_output=True,
                text=True,
            )

            for formula in json.loads(brew_info_result.stdout)["formulae"]:
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


def run_checked(cmd: list[str], out: list[str], err: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    out.append(result.stdout)
    err.append(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def download(url: str, expected_hash: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {expected_hash}, got {digest}"
        )


def install_brew_packages(out: list[str], err: list[str]) -> None:
    run_checked(["brew", "install", "--display-times", *PACKAGES], out, err)


def install_font(out: list[str], err: list[str], font: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "font.zip"
        download(font["url"], font["hash"], archive)

        extracted = tmp_path / "extracted"
        run_checked(["unzip", "-o", str(archive), "-d", str(extracted)], out, err)

        font_dir = Path.home() / "Library" / "Fonts"
        font_dir.mkdir(parents=True, exist_ok=True)
        for path in extracted.rglob("*"):
            if path.suffix.lower() in {".ttc", ".ttf", ".otf"}:
                shutil.copy2(path, font_dir / path.name)
                out.append(f"Installed {path.name}\n")


def install_inkscape(out: list[str], err: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dmg = tmp_path / "Inkscape.dmg"
        mount_point = tmp_path / "mnt"
        mount_point.mkdir()
        download(INKSCAPE["url"], INKSCAPE["hash"], dmg)

        run_checked(
            ["hdiutil", "attach", "-nobrowse", "-readonly",
             "-mountpoint", str(mount_point), str(dmg)],
            out, err,
        )
        try:
            run_checked(
                ["ditto", str(mount_point / "Inkscape.app"),
                 "/Applications/Inkscape.app"],
                out, err,
            )
        finally:
            run_checked(["hdiutil", "detach", str(mount_point)], out, err)


def run_task(func, *args) -> tuple[int, str, str]:
    out: list[str] = []
    err: list[str] = []
    try:
        func(out, err, *args)
        code = 0
    except subprocess.CalledProcessError as e:
        code = e.returncode
    except Exception as e:
        err.append(f"{e}\n")
        code = 1
    return code, "".join(out), "".join(err)


def main():
    if len(sys.argv) < 2:
        exit("No command provided")

    command = sys.argv[1]
    if command == "clean":
        # `sudo mkdir -p /opt/homebrew-aside/Cellar`
        # `sudo mkdir -p /opt/homebrew-aside/Caskroom`
        # `sudo mv /opt/homebrew/.git /opt/homebrew-aside`
        # `sudo mv /opt/homebrew/Cellar/* /opt/homebrew-aside/Cellar`
        # `sudo mv /opt/homebrew/Caskroom/* /opt/homebrew-aside/Caskroom`
        # `time find /opt/homebrew -maxdepth 2 -type l -delete`
        aside = "/opt/homebrew-aside"
        brew_root = Path("/opt/homebrew")
        subprocess.run(["sudo", "mkdir", "-p", f"{aside}/Cellar"], check=True)
        subprocess.run(["sudo", "mkdir", "-p", f"{aside}/Caskroom"], check=True)
        subprocess.run(["sudo", "mv", str(brew_root / ".git"), aside], check=True)
        for sub in ("Cellar", "Caskroom"):
            entries = [str(p) for p in (brew_root / sub).glob("*")]
            if entries:
                subprocess.run(
                    ["sudo", "mv", *entries, f"{aside}/{sub}"], check=True
                )
        start = time.perf_counter()
        subprocess.run(
            ["find", str(brew_root), "-maxdepth", "2", "-type", "l", "-delete"],
            check=True,
        )
        print(f"find took {time.perf_counter() - start:.2f}s")

    elif command == "make-cache":
        package_filter = PackageFilter(PACKAGES, CACHE_EXCLUDE)
        filtered_packages = package_filter.get_filtered_packages()
        filtered_packages = sorted(set(filtered_packages) | set(CACHE_INCLUDE))
        subprocess.run(
            ["brew", "install", "--display-times", *filtered_packages], check=True
        )

    elif command == "install":
        tasks = {
            "brew install": (install_brew_packages,),
            "font 0": (install_font, FONTS[0]),
            "font 1": (install_font, FONTS[1]),
            "inkscape": (install_inkscape,),
        }
        exit_code = 0
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {
                executor.submit(run_task, *task): name
                for name, task in tasks.items()
            }
            for future in as_completed(futures):
                code, out, err, elapsed = future.result()
                print(f"::group::{futures[future]} (exit {code}, {elapsed:.2f}s)")
                if out:
                    print(out, end="")
                if err:
                    print(err, end="")
                print("::endgroup::")
                sys.stdout.flush()
                exit_code = exit_code or code
        if exit_code:
            sys.exit(exit_code)

    else:
        exit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
