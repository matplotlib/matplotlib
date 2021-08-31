import os
import subprocess

from .file_finder import scm_find_files
from .file_finder import is_toplevel_acceptable


def _hg_toplevel(path):
    try:
        with open(os.devnull, "wb") as devnull:
            out = subprocess.check_output(
                ["hg", "root"],
                cwd=(path or "."),
                universal_newlines=True,
                stderr=devnull,
            )
        return os.path.normcase(os.path.realpath(out.strip()))
    except subprocess.CalledProcessError:
        # hg returned error, we are not in a mercurial repo
        return None
    except OSError:
        # hg command not found, probably
        return None


def _hg_ls_files_and_dirs(toplevel):
    hg_files = set()
    hg_dirs = {toplevel}
    out = subprocess.check_output(
        ["hg", "files"], cwd=toplevel, universal_newlines=True
    )
    for name in out.splitlines():
        name = os.path.normcase(name).replace("/", os.path.sep)
        fullname = os.path.join(toplevel, name)
        hg_files.add(fullname)
        dirname = os.path.dirname(fullname)
        while len(dirname) > len(toplevel) and dirname not in hg_dirs:
            hg_dirs.add(dirname)
            dirname = os.path.dirname(dirname)
    return hg_files, hg_dirs


def hg_find_files(path=""):
    toplevel = _hg_toplevel(path)
    if not is_toplevel_acceptable(toplevel):
        return []
    hg_files, hg_dirs = _hg_ls_files_and_dirs(toplevel)
    return scm_find_files(path, hg_files, hg_dirs)
