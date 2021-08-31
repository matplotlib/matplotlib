"""
utils
"""
import inspect
import warnings
import sys
import shlex
import subprocess
import os
import platform
import traceback


DEBUG = bool(os.environ.get("SETUPTOOLS_SCM_DEBUG"))
IS_WINDOWS = platform.system() == "Windows"


def no_git_env(env):
    # adapted from pre-commit
    # Too many bugs dealing with environment variables and GIT:
    # https://github.com/pre-commit/pre-commit/issues/300
    # In git 2.6.3 (maybe others), git exports GIT_WORK_TREE while running
    # pre-commit hooks
    # In git 1.9.1 (maybe others), git exports GIT_DIR and GIT_INDEX_FILE
    # while running pre-commit hooks in submodules.
    # GIT_DIR: Causes git clone to clone wrong thing
    # GIT_INDEX_FILE: Causes 'error invalid object ...' during commit
    for k, v in env.items():
        if k.startswith("GIT_"):
            trace(k, v)
    return {
        k: v
        for k, v in env.items()
        if not k.startswith("GIT_")
        or k in ("GIT_EXEC_PATH", "GIT_SSH", "GIT_SSH_COMMAND")
    }


def trace(*k):
    if DEBUG:
        print(*k)
        sys.stdout.flush()


def trace_exception():
    DEBUG and traceback.print_exc()


def ensure_stripped_str(str_or_bytes):
    if isinstance(str_or_bytes, str):
        return str_or_bytes.strip()
    else:
        return str_or_bytes.decode("utf-8", "surrogateescape").strip()


def _always_strings(env_dict):
    """
    On Windows and Python 2, environment dictionaries must be strings
    and not unicode.
    """
    if IS_WINDOWS:
        env_dict.update((key, str(value)) for (key, value) in env_dict.items())
    return env_dict


def _popen_pipes(cmd, cwd):
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        env=_always_strings(
            dict(
                no_git_env(os.environ),
                # os.environ,
                # try to disable i18n
                LC_ALL="C",
                LANGUAGE="",
                HGPLAIN="1",
            )
        ),
    )


def do_ex(cmd, cwd="."):
    trace("cmd", repr(cmd))
    if os.name == "posix" and not isinstance(cmd, (list, tuple)):
        cmd = shlex.split(cmd)

    p = _popen_pipes(cmd, cwd)
    out, err = p.communicate()
    if out:
        trace("out", repr(out))
    if err:
        trace("err", repr(err))
    if p.returncode:
        trace("ret", p.returncode)
    return ensure_stripped_str(out), ensure_stripped_str(err), p.returncode


def do(cmd, cwd="."):
    out, err, ret = do_ex(cmd, cwd)
    if ret:
        print(err)
    return out


def data_from_mime(path):
    with open(path, encoding="utf-8") as fp:
        content = fp.read()
    trace("content", repr(content))
    # the complex conditions come from reading pseudo-mime-messages
    data = dict(x.split(": ", 1) for x in content.splitlines() if ": " in x)
    trace("data", data)
    return data


def function_has_arg(fn, argname):
    assert inspect.isfunction(fn)

    argspec = inspect.signature(fn).parameters

    return argname in argspec


def has_command(name, warn=True):
    try:
        p = _popen_pipes([name, "help"], ".")
    except OSError:
        trace(*sys.exc_info())
        res = False
    else:
        p.communicate()
        res = not p.returncode
    if not res and warn:
        warnings.warn("%r was not found" % name, category=RuntimeWarning)
    return res


def require_command(name):
    if not has_command(name, warn=False):
        raise OSError("%r was not found" % name)
