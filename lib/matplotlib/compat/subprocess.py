"""
A replacement wrapper around the subprocess module, with a number of
work-arounds:
- Provides the check_output function (which subprocess only provides from Python
  2.7 onwards).
- Provides a stub implementation of subprocess members on Google App Engine
  (which are missing in subprocess).

Instead of importing subprocess, other modules should use this as follows:

from matplotlib.compat import subprocess

This module is safe to import from anywhere within matplotlib.
"""

from __future__ import absolute_import    # Required to import subprocess
from __future__ import print_function

import subprocess

__all__ = ['Popen', 'PIPE', 'STDOUT', 'check_output']


if hasattr(subprocess, 'Popen'):
    Popen = subprocess.Popen
    # Assume that it also has the other constants.
    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
else:
    # In restricted environments (such as Google App Engine), these are
    # non-existent. Replace them with dummy versions that always raise OSError.
    def Popen(*args, **kwargs):
        raise OSError("subprocess.Popen is not supported")
    PIPE = -1
    STDOUT = -2


def _check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte
    string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the
    returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example::

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.::

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    """
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = Popen(stdout=PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output


# python2.7's subprocess provides a check_output method
if hasattr(subprocess, 'check_output'):
    check_output = subprocess.check_output
else:
    check_output = _check_output

CalledProcessError = subprocess.CalledProcessError
