# Usage: python3 deps.py "ccache ffmpeg ..." "python icu4c"
#
# Prints the dependencies of the given packages, minus the excluded ones, in a
# form suitable for passing to `brew install`. Not run or tested. It relies on
# `brew info --json=v2` reporting direct runtime dependencies (build and test
# dependencies are in separate fields) under the key "dependencies".

import json
import subprocess
import sys


def info(names):
    # One `brew info` call for a whole list of formulae. --json=v2 gives a
    # single JSON object with a "formulae" array holding one entry per name.
    # check=True raises CalledProcessError if brew fails (e.g. unknown name).
    result = subprocess.run(
        ["brew", "info", "--json=v2", "--formula", *names],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["formulae"]


def main():
    # First argument: space-separated packages. Second: space-separated
    # excludes. split() with no argument also handles repeated whitespace and
    # an empty string (which yields an empty list).
    packages = sys.argv[1].split()
    excludes = set(sys.argv[2].split())

    def is_excluded(name):
        # Versioned formulae look like "python@3.13" or "icu4c@78". Cutting
        # at the first "@" lets the plain exclude "python" match all of them.
        return name.split("@")[0] in excludes

    # graph maps a formula name to the list of its direct runtime
    # dependencies. It is filled in one layer at a time below.
    graph = {}

    # requested records every name already sent to `brew info`, so no formula
    # is fetched twice. pending is the next layer to fetch.
    requested = set(packages)
    pending = list(packages)

    while pending:
        # Fetch the whole layer in a single call instead of one per formula.
        for formula in info(pending):
            deps = formula.get("dependencies", [])
            # Dependencies of tap formulae may be listed by full name
            # ("user/tap/name"), so the entry is stored under both names to
            # make later lookups succeed either way.
            graph[formula["name"]] = deps
            graph[formula["full_name"]] = deps

        # The next layer is every dependency seen so far that has not been
        # requested yet. Excluded names are skipped, so their own
        # dependencies are never fetched either. Sorting makes the order of
        # `brew info` calls deterministic.
        pending = sorted(
            {
                dep
                for deps in graph.values()
                for dep in deps
                if not is_excluded(dep) and dep not in requested
            }
        )
        requested.update(pending)

    # memo caches the result of tainted() per formula. A formula is "tainted"
    # when installing it would pull in an excluded formula, directly or
    # through any chain of dependencies. brew install resolves dependencies
    # itself, so a tainted formula cannot be installed without installing the
    # excluded one. The only way around that is to leave it out entirely.
    memo = {}

    def tainted(name):
        if name in memo:
            return memo[name]
        # Mark the formula as untainted before recursing. If the dependency
        # graph ever contains a cycle, the recursion then stops here instead
        # of looping forever.
        memo[name] = False
        memo[name] = any(
            is_excluded(dep) or tainted(dep) for dep in graph.get(name, [])
        )
        return memo[name]

    # keep is the final set of formulae to install. The traversal starts from
    # the requested packages themselves. The packages are never added to
    # keep, because the goal is to install only their dependencies.
    keep = set()
    stack = list(packages)
    while stack:
        name = stack.pop()
        for dep in graph.get(name, []):
            # Skip excluded formulae, tainted formulae, and anything already
            # kept. Skipping a tainted formula also skips everything below
            # it, since the traversal never descends into it.
            if is_excluded(dep) or tainted(dep) or dep in keep:
                continue
            keep.add(dep)
            stack.append(dep)

    # Print one line of space-separated names, sorted so that the output is
    # stable between runs.
    print(" ".join(sorted(keep)))


if __name__ == "__main__":
    main()
