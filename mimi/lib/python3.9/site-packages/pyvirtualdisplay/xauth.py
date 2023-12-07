"""Utility functions for xauth."""
import hashlib
import os
import subprocess


class NotFoundError(Exception):
    """Error when xauth was not found."""


def is_installed():
    """
    Return whether or not xauth is installed.
    """
    try:
        xauth = subprocess.Popen(
            ["xauth", "-V"],
            # env=self._env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, _ = xauth.communicate()
        # p = EasyProcess(["xauth", "-V"])
        # p.enable_stdout_log = False
        # p.enable_stderr_log = False
        # p.call()
    except FileNotFoundError:
        return False
    else:
        return True


def generate_mcookie():
    """
    Generate a cookie string suitable for xauth.
    """
    data = os.urandom(16)  # 16 bytes = 128 bit
    return hashlib.md5(data).hexdigest()


def call(*args):
    """
    Call xauth with the given args.
    """
    xauth = subprocess.Popen(
        ["xauth"] + list(args),
        # env=self._env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, _ = xauth.communicate()
    # EasyProcess(["xauth"] + list(args)).call()
