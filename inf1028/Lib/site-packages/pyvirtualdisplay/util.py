import subprocess
import sys


def get_helptext(program):
    cmd = [program, "-help"]

    # py3.7+
    # p = subprocess.run(cmd, capture_output=True)
    # stderr = p.stderr

    # py3.6 also
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    _, stderr = p.communicate()

    helptext = stderr.decode("utf-8", "ignore")
    return helptext


def platform_is_osx():
    return sys.platform == "darwin"
