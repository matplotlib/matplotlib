from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard
    from . import _types as _t

from .utils import trace


def scm_find_files(
    path: _t.PathT, scm_files: set[str], scm_dirs: set[str]
) -> list[str]:
    """ setuptools compatible file finder that follows symlinks

    - path: the root directory from which to search
    - scm_files: set of scm controlled files and symlinks
      (including symlinks to directories)
    - scm_dirs: set of scm controlled directories
      (including directories containing no scm controlled files)

    scm_files and scm_dirs must be absolute with symlinks resolved (realpath),
    with normalized case (normcase)

    Spec here: http://setuptools.readthedocs.io/en/latest/setuptools.html#\
        adding-support-for-revision-control-systems
    """
    realpath = os.path.normcase(os.path.realpath(path))
    seen: set[str] = set()
    res: list[str] = []
    for dirpath, dirnames, filenames in os.walk(realpath, followlinks=True):
        # dirpath with symlinks resolved
        realdirpath = os.path.normcase(os.path.realpath(dirpath))

        def _link_not_in_scm(n: str) -> bool:
            fn = os.path.join(realdirpath, os.path.normcase(n))
            return os.path.islink(fn) and fn not in scm_files

        if realdirpath not in scm_dirs:
            # directory not in scm, don't walk it's content
            dirnames[:] = []
            continue
        if os.path.islink(dirpath) and not os.path.relpath(
            realdirpath, realpath
        ).startswith(os.pardir):
            # a symlink to a directory not outside path:
            # we keep it in the result and don't walk its content
            res.append(os.path.join(path, os.path.relpath(dirpath, path)))
            dirnames[:] = []
            continue
        if realdirpath in seen:
            # symlink loop protection
            dirnames[:] = []
            continue
        dirnames[:] = [dn for dn in dirnames if not _link_not_in_scm(dn)]
        for filename in filenames:
            if _link_not_in_scm(filename):
                continue
            # dirpath + filename with symlinks preserved
            fullfilename = os.path.join(dirpath, filename)
            if os.path.normcase(os.path.realpath(fullfilename)) in scm_files:
                res.append(os.path.join(path, os.path.relpath(fullfilename, realpath)))
        seen.add(realdirpath)
    return res


def is_toplevel_acceptable(toplevel: str | None) -> TypeGuard[str]:
    """ """
    if toplevel is None:
        return False

    ignored = os.environ.get("SETUPTOOLS_SCM_IGNORE_VCS_ROOTS", "").split(os.pathsep)
    ignored = [os.path.normcase(p) for p in ignored]

    trace(toplevel, ignored)

    return toplevel not in ignored
