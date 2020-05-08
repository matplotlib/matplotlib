"""
Run and time some or all examples.
"""

from argparse import ArgumentParser
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


_preamble = """\
from matplotlib import pyplot as plt

def pseudo_show(block=True):
    for num in plt.get_fignums():
        plt.figure(num).savefig(f"{num}")

plt.show = pseudo_show

"""


class RunInfo:
    def __init__(self, backend, elapsed, failed):
        self.backend = backend
        self.elapsed = elapsed
        self.failed = failed

    def __str__(self):
        s = ""
        if self.backend:
            s += f"{self.backend}: "
        s += f"{self.elapsed}ms"
        if self.failed:
            s += " (failed!)"
        return s


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", action="append",
        help=("backend to test; can be passed multiple times; defaults to the "
              "default backend"))
    parser.add_argument(
        "--include-sgskip", action="store_true",
        help="do not filter out *_sgskip.py examples")
    parser.add_argument(
        "--rundir", type=Path,
        help=("directory from where the tests are run; defaults to a "
              "temporary directory"))
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="examples to run; defaults to all examples (except *_sgskip.py)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "examples"
    paths = args.paths if args.paths else sorted(root.glob("**/*.py"))
    if not args.include_sgskip:
        paths = [path for path in paths if not path.stem.endswith("sgskip")]
    relpaths = [path.resolve().relative_to(root) for path in paths]
    width = max(len(str(relpath)) for relpath in relpaths)
    for relpath in relpaths:
        print(str(relpath).ljust(width + 1), end="", flush=True)
        runinfos = []
        with ExitStack() as stack:
            if args.rundir:
                cwd = args.rundir / relpath.with_suffix("")
                cwd.mkdir(parents=True)
            else:
                cwd = stack.enter_context(TemporaryDirectory())
            Path(cwd, relpath.name).write_text(
                _preamble + (root / relpath).read_text())
            for backend in args.backend or [None]:
                env = {**os.environ}
                if backend is not None:
                    env["MPLBACKEND"] = backend
                start = time.perf_counter()
                proc = subprocess.run([sys.executable, relpath.name],
                                      cwd=cwd, env=env)
                elapsed = round(1000 * (time.perf_counter() - start))
                runinfos.append(RunInfo(backend, elapsed, proc.returncode))
        print("\t".join(map(str, runinfos)))


if __name__ == "__main__":
    main()
