# Usage: python3 macos_brew_cache.py "ccache ffmpeg ..." "python icu4c"
#
# Prints the recursive dependencies of the given packages, minus the excluded ones
#

import json
import subprocess
import sys

from collections.abc import Sequence


class Graph:
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

    def get_filtered_packages(self) -> set[str]:
        result = set()
        stack = list(self._packages)

        while stack:
            name = self._resolve_name(stack.pop())
            for dep in self._dependencies.get(name, []):
                if dep in result:
                    continue
                if not self.is_excluded(dep) and not self.is_tainted(dep):
                    result.add(dep)
                stack.append(dep)

        return sorted(result)


def main():
    argv = sys.argv
    packages = argv[1].split() if len(argv) > 1 and argv[1] else []
    excludes = argv[2].split() if len(argv) > 2 and argv[2] else []

    graph = Graph(packages, excludes)

    print(" ".join(graph.get_filtered_packages()))

if __name__ == "__main__":
    main()
