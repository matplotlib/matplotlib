from __future__ import annotations

import logging
import os
import subprocess
import tarfile
from typing import IO
from typing import TYPE_CHECKING

from .file_finder import is_toplevel_acceptable
from .file_finder import scm_find_files
from .utils import do_ex
from .utils import trace

if TYPE_CHECKING:
    from . import _types as _t


log = logging.getLogger(__name__)


def _git_toplevel(path: str) -> str | None:
    try:
        cwd = os.path.abspath(path or ".")
        out, err, ret = do_ex(["git", "rev-parse", "HEAD"], cwd=cwd)
        if ret != 0:
            # BAIL if there is no commit
            log.error("listing git files failed - pretending there aren't any")
            return None
        out, err, ret = do_ex(
            ["git", "rev-parse", "--show-prefix"],
            cwd=cwd,
        )
        if ret != 0:
            return None
        out = out.strip()[:-1]  # remove the trailing pathsep
        if not out:
            out = cwd
        else:
            # Here, ``out`` is a relative path to root of git.
            # ``cwd`` is absolute path to current working directory.
            # the below method removes the length of ``out`` from
            # ``cwd``, which gives the git toplevel
            assert cwd.replace("\\", "/").endswith(out), f"cwd={cwd!r}\nout={out!r}"
            # In windows cwd contains ``\`` which should be replaced by ``/``
            # for this assertion to work. Length of string isn't changed by replace
            # ``\\`` is just and escape for `\`
            out = cwd[: -len(out)]
        trace("find files toplevel", out)
        return os.path.normcase(os.path.realpath(out.strip()))
    except subprocess.CalledProcessError:
        # git returned error, we are not in a git repo
        return None
    except OSError:
        # git command not found, probably
        return None


def _git_interpret_archive(fd: IO[bytes], toplevel: str) -> tuple[set[str], set[str]]:
    with tarfile.open(fileobj=fd, mode="r|*") as tf:
        git_files = set()
        git_dirs = {toplevel}
        for member in tf.getmembers():
            name = os.path.normcase(member.name).replace("/", os.path.sep)
            if member.type == tarfile.DIRTYPE:
                git_dirs.add(name)
            else:
                git_files.add(name)
        return git_files, git_dirs


def _git_ls_files_and_dirs(toplevel: str) -> tuple[set[str], set[str]]:
    # use git archive instead of git ls-file to honor
    # export-ignore git attribute

    cmd = ["git", "archive", "--prefix", toplevel + os.path.sep, "HEAD"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, cwd=toplevel, stderr=subprocess.DEVNULL
    )
    assert proc.stdout is not None
    try:
        try:
            return _git_interpret_archive(proc.stdout, toplevel)
        finally:
            # ensure we avoid resource warnings by cleaning up the process
            proc.stdout.close()
            proc.terminate()
    except Exception:
        if proc.wait() != 0:
            log.error("listing git files failed - pretending there aren't any")
        return set(), set()


def git_find_files(path: _t.PathT = "") -> list[str]:
    toplevel = _git_toplevel(os.fspath(path))
    if not is_toplevel_acceptable(toplevel):
        return []
    assert toplevel is not None  # mypy ignores typeguard
    fullpath = os.path.abspath(os.path.normpath(path))
    if not fullpath.startswith(toplevel):
        trace("toplevel mismatch", toplevel, fullpath)
    git_files, git_dirs = _git_ls_files_and_dirs(toplevel)
    return scm_find_files(path, git_files, git_dirs)
