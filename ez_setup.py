#!python
"""Bootstrap setuptools installation

If you want to use setuptools in your package's setup.py, just include this
file in the same directory with it, and add this to the top of your setup.py::

    from ez_setup import use_setuptools
    use_setuptools()

If you want to require a specific version of setuptools, set a download
mirror, or use an alternate download directory, you can do so by supplying
the appropriate options to ``use_setuptools()``.

This file can also be run as a script to install or upgrade setuptools.
"""
import os
import shutil
import sys
import tempfile
import tarfile
import optparse
import subprocess

from distutils import log

try:
    from site import USER_SITE
except ImportError:
    USER_SITE = None

DEFAULT_VERSION = "0.7.2"
DEFAULT_URL = "https://pypi.python.org/packages/source/s/setuptools/"

def _python_cmd(*args):
    args = (sys.executable,) + args
    return subprocess.call(args) == 0

def _install(tarball, install_args=()):
    # extracting the tarball
    tmpdir = tempfile.mkdtemp()
    log.warn('Extracting in %s', tmpdir)
    old_wd = os.getcwd()
    try:
        os.chdir(tmpdir)
        tar = tarfile.open(tarball)
        _extractall(tar)
        tar.close()

        # going in the directory
        subdir = os.path.join(tmpdir, os.listdir(tmpdir)[0])
        os.chdir(subdir)
        log.warn('Now working in %s', subdir)

        # installing
        log.warn('Installing Setuptools')
        if not _python_cmd('setup.py', 'install', *install_args):
            log.warn('Something went wrong during the installation.')
            log.warn('See the error message above.')
            # exitcode will be 2
            return 2
    finally:
        os.chdir(old_wd)
        shutil.rmtree(tmpdir)


def _build_egg(egg, tarball, to_dir):
    # extracting the tarball
    tmpdir = tempfile.mkdtemp()
    log.warn('Extracting in %s', tmpdir)
    old_wd = os.getcwd()
    try:
        os.chdir(tmpdir)
        tar = tarfile.open(tarball)
        _extractall(tar)
        tar.close()

        # going in the directory
        subdir = os.path.join(tmpdir, os.listdir(tmpdir)[0])
        os.chdir(subdir)
        log.warn('Now working in %s', subdir)

        # building an egg
        log.warn('Building a Setuptools egg in %s', to_dir)
        _python_cmd('setup.py', '-q', 'bdist_egg', '--dist-dir', to_dir)

    finally:
        os.chdir(old_wd)
        shutil.rmtree(tmpdir)
    # returning the result
    log.warn(egg)
    if not os.path.exists(egg):
        raise IOError('Could not build the egg.')


def _do_download(version, download_base, to_dir, download_delay):
    egg = os.path.join(to_dir, 'setuptools-%s-py%d.%d.egg'
                       % (version, sys.version_info[0], sys.version_info[1]))
    if not os.path.exists(egg):
        tarball = download_setuptools(version, download_base,
                                      to_dir, download_delay)
        _build_egg(egg, tarball, to_dir)
    sys.path.insert(0, egg)
    import setuptools
    setuptools.bootstrap_install_from = egg


def use_setuptools(version=DEFAULT_VERSION, download_base=DEFAULT_URL,
                   to_dir=os.curdir, download_delay=15):
    # making sure we use the absolute path
    to_dir = os.path.abspath(to_dir)
    was_imported = 'pkg_resources' in sys.modules or \
        'setuptools' in sys.modules
    try:
        import pkg_resources
    except ImportError:
        return _do_download(version, download_base, to_dir, download_delay)
    try:
        pkg_resources.require("setuptools>=" + version)
        return
    except pkg_resources.VersionConflict:
        e = sys.exc_info()[1]
        if was_imported:
            sys.stderr.write(
            "The required version of setuptools (>=%s) is not available,\n"
            "and can't be installed while this script is running. Please\n"
            "install a more recent version first, using\n"
            "'easy_install -U setuptools'."
            "\n\n(Currently using %r)\n" % (version, e.args[0]))
            sys.exit(2)
        else:
            del pkg_resources, sys.modules['pkg_resources']    # reload ok
            return _do_download(version, download_base, to_dir,
                                download_delay)
    except pkg_resources.DistributionNotFound:
        return _do_download(version, download_base, to_dir,
                            download_delay)


def download_setuptools(version=DEFAULT_VERSION, download_base=DEFAULT_URL,
                        to_dir=os.curdir, delay=15):
    """Download setuptools from a specified location and return its filename

    `version` should be a valid setuptools version number that is available
    as an egg for download under the `download_base` URL (which should end
    with a '/'). `to_dir` is the directory where the egg will be downloaded.
    `delay` is the number of seconds to pause before an actual download
    attempt.
    """
    # making sure we use the absolute path
    to_dir = os.path.abspath(to_dir)
    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib2 import urlopen
    tgz_name = "setuptools-%s.tar.gz" % version
    url = download_base + tgz_name
    saveto = os.path.join(to_dir, tgz_name)
    src = dst = None
    if not os.path.exists(saveto):  # Avoid repeated downloads
        try:
            log.warn("Downloading %s", url)
            src = urlopen(url)
            # Read/write all in one block, so we don't create a corrupt file
            # if the download is interrupted.
            data = src.read()
            dst = open(saveto, "wb")
            dst.write(data)
        finally:
            if src:
                src.close()
            if dst:
                dst.close()
    return os.path.realpath(saveto)


def _extractall(self, path=".", members=None):
    """Extract all members from the archive to the current working
       directory and set owner, modification time and permissions on
       directories afterwards. `path' specifies a different directory
       to extract to. `members' is optional and must be a subset of the
       list returned by getmembers().
    """
    import copy
    import operator
    from tarfile import ExtractError
    directories = []

    if members is None:
        members = self

    for tarinfo in members:
        if tarinfo.isdir():
            # Extract directories with a safe mode.
            directories.append(tarinfo)
            tarinfo = copy.copy(tarinfo)
            tarinfo.mode = 448  # decimal for oct 0700
        self.extract(tarinfo, path)

    # Reverse sort directories.
    if sys.version_info < (2, 4):
        def sorter(dir1, dir2):
            return cmp(dir1.name, dir2.name)
        directories.sort(sorter)
        directories.reverse()
    else:
        directories.sort(key=operator.attrgetter('name'), reverse=True)

    # Set correct owner, mtime and filemode on directories.
    for tarinfo in directories:
        dirpath = os.path.join(path, tarinfo.name)
        try:
            self.chown(tarinfo, dirpath)
            self.utime(tarinfo, dirpath)
            self.chmod(tarinfo, dirpath)
        except ExtractError:
            e = sys.exc_info()[1]
            if self.errorlevel > 1:
                raise
            else:
                self._dbg(1, "tarfile: %s" % e)


def _build_install_args(options):
    """
    Build the arguments to 'python setup.py install' on the setuptools package
    """
    install_args = []
    if options.user_install:
        if sys.version_info < (2, 6):
            log.warn("--user requires Python 2.6 or later")
            raise SystemExit(1)
        install_args.append('--user')
    return install_args

def _parse_args():
    """
    Parse the command line for options
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '--user', dest='user_install', action='store_true', default=False,
        help='install in user site package (requires Python 2.6 or later)')
    parser.add_option(
        '--download-base', dest='download_base', metavar="URL",
        default=DEFAULT_URL,
        help='alternative URL from where to download the setuptools package')
    options, args = parser.parse_args()
    # positional arguments are ignored
    return options

def main(version=DEFAULT_VERSION):
    """Install or upgrade setuptools and EasyInstall"""
    options = _parse_args()
    tarball = download_setuptools(download_base=options.download_base)
    return _install(tarball, _build_install_args(options))

if __name__ == '__main__':
    sys.exit(main())
