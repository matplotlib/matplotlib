from importlib import machinery
import os
import shutil
import sys

# These are modules which otherwise *should* be removed, but can't since they
# are requirements of pytest.

# TODO: Find a solution to this, possibly involving running tests in a
# separate, more limited process.

PYTEST_REQUIREMENTS = [
    '_posixsubprocess', 'select', 'binascii', 'pyexpat', '_socket'
]

# These are optional parts of the standard library that are required for core
# functionality in matplotlib that can not easily be avoided.

MATPLOTLIB_HARD_REQUIREMENTS = [
    'zlib', '_csv'
]


KEEP_MODULES = PYTEST_REQUIREMENTS + MATPLOTLIB_HARD_REQUIREMENTS


def iterate_modules_to_remove():
    with open(os.path.join(
            os.path.dirname(__file__), 'Setup.dist'), 'r') as fd:
        for line in fd:
            if line.startswith('# Modules with some UNIX dependencies'):
                break

        for line in fd:
            line = line.strip()
            if line.startswith('#') and not line.startswith('# ') and len(line) > 1:
                modulename = line[1:].split()[0]
                if modulename not in KEEP_MODULES:
                    yield modulename

    # Some other commonly-missing modules that aren't in Setup.dist
    yield '_bz2'
    yield '_ctypes'
    yield '_lzma'
    yield '_lsprof'
    yield '_multiprocessing'
    yield 'ossaudiodev'
    yield '_tkinter'
    yield 'tkinter'
    yield '_sqlite3'
    yield '_thread'


def main():
    PYTHON_VERSION = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    DST_ROOT = os.path.abspath('./minimal_python')
    LIB_ROOT = os.path.join(DST_ROOT, 'lib', PYTHON_VERSION)
    shutil.rmtree(os.path.join(LIB_ROOT, '__pycache__'))

    SETUP_DIST_PATH = os.path.join(os.path.dirname(__file__), 'Setup.dist')
    for modulename in iterate_modules_to_remove():
        for root in ['', 'lib-dynload']:
            for ext in machinery.all_suffixes():
                modulepath = os.path.join(LIB_ROOT, root, modulename + ext)
                if os.path.isfile(modulepath):
                    os.remove(modulepath)
                elif os.path.isdir(modulepath):
                    os.rmtree(modulepath)


if __name__ == '__main__':
    main()
