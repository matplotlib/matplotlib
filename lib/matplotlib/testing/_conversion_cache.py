"""
A cache of png files keyed by the MD5 hashes of corresponding svg and
pdf files, to reduce test suite running times for svg and pdf files
that stay exactly the same from one run to the next.

There is a corresponding nose plugin in testing/nose/plugins and
similar pytest code in conftest.py.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import hashlib
import shutil
import os
import warnings

from matplotlib import _get_cachedir
from matplotlib import cbook
from matplotlib import checkdep_ghostscript
from matplotlib import checkdep_inkscape


class ConversionCache(object):
    """A cache that stores png files converted from svg or pdf formats.

    The image comparison test cases compare svg and pdf files by
    converting them to png files. When a test case has produced a
    file, e.g. result.pdf, it queries this cache by the pathname
    '/path/to/result_images/result.pdf'. The cache computes a hash of
    the file (and the version of the external software used to convert
    the file) and if a result by that hash value is available, writes
    the data to the output location supplied by the caller. Otherwise
    the test case has to run the conversion and can then insert the
    result into the cache.

    Parameters
    ----------
    directory : str, optional
        Files are stored in this directory, defaults to `'test_cache'` in
        the overall Matplotlib cache directory.
    max_size : int, optional
        The flush method will delete files until their combined size is
        under this limit, in bytes. Defaults to 100 megabytes.

    """

    def __init__(self, directory=None, max_size=int(1e8)):
        self.gets = set()
        self.hits = set()
        if directory is not None:
            self.cachedir = directory
        else:
            self.cachedir = self.get_cache_dir()
        self.ensure_cache_dir()
        if not isinstance(max_size, int):
            raise ValueError("max_size is %s, expected int" % type(max_size))
        self.max_size = max_size
        self.cached_ext = '.png'
        self.converter_version = {}
        try:
            self.converter_version['.pdf'] = \
                checkdep_ghostscript()[1].encode('utf-8')
        except:
            pass
        try:
            self.converter_version['.svg'] = \
                checkdep_inkscape().encode('utf-8')
        except:
            pass
        self.hash_cache = {}

    def get(self, filename, newname):
        """Query the cache.

        Parameters
        ----------
        filename : str
            Full path to the original file.
        newname : str
            Path to which the result should be written.

        Returns
        -------
        bool
            True if the file was found in the cache and is now written
            to `newname`.
        """
        self.gets.add(filename)
        hash_value = self._get_file_hash(filename)
        cached_file = os.path.join(self.cachedir, hash_value + self.cached_ext)
        with cbook.Locked(self.cachedir):
            if os.path.exists(cached_file):
                shutil.copyfile(cached_file, newname)
                self.hits.add(filename)
                return True
            else:
                return False

    def put(self, original, converted):
        """Insert a file into the cache.

        Parameters
        ----------
        original : str
            Full path to the original file.
        converted : str
            Full path to the png file converted from the original.
        """
        hash_value = self._get_file_hash(original)
        cached_file = os.path.join(self.cachedir, hash_value + self.cached_ext)
        with cbook.Locked(self.cachedir):
            shutil.copyfile(converted, cached_file)

    def _get_file_hash(self, path, block_size=2 ** 20):
        if path in self.hash_cache:
            return self.hash_cache[path]
        _, ext = os.path.splitext(path)
        version_tag = self.converter_version.get(ext)
        if version_tag is None:
            warnings.warn(
                ("Don't know the external converter for files with extension "
                 "%s, cannot ensure cache invalidation on version update.")
                % ext)
        result = self._get_file_hash_static(path, block_size, version_tag)
        self.hash_cache[path] = result
        return result

    @staticmethod
    def _get_file_hash_static(path, block_size, version_tag):
        # the parts of _get_file_hash that are called from the deprecated
        # compare.get_file_hash; can merge into _get_file_hash once that
        # function is removed
        md5 = hashlib.md5()
        with open(path, 'rb') as fd:
            while True:
                data = fd.read(block_size)
                if not data:
                    break
                md5.update(data)
        if version_tag is not None:
            md5.update(version_tag)
        return md5.hexdigest()

    def report(self):
        """Return information about the cache.

        Returns
        -------
        r : dict
            `r['gets']` is the set of files queried,
            `r['hits']` is the set of files found in the cache
        """
        return dict(hits=self.hits, gets=self.gets)

    def expire(self):
        """Delete cached files until the disk usage is under the limit.

        Orders files by access time, so the least recently used files
        get deleted first.
        """
        with cbook.Locked(self.cachedir):
            stats = {filename: os.stat(os.path.join(self.cachedir, filename))
                     for filename in os.listdir(self.cachedir)
                     if filename.endswith(self.cached_ext)}
            usage = sum(f.st_size for f in stats.values())
            to_free = usage - self.max_size
            if to_free <= 0:
                return

            files = sorted(stats.keys(),
                           key=lambda f: stats[f].st_atime,
                           reverse=True)
            while to_free > 0:
                filename = files.pop()
                os.remove(os.path.join(self.cachedir, filename))
                to_free -= stats[filename].st_size

    @staticmethod
    def get_cache_dir():
        cachedir = _get_cachedir()
        if cachedir is None:
            raise CacheError('No suitable configuration directory')
        cachedir = os.path.join(cachedir, 'test_cache')
        return cachedir

    def ensure_cache_dir(self):
        if not os.path.exists(self.cachedir):
            try:
                cbook.mkdirs(self.cachedir)
            except IOError as e:
                raise CacheError("Error creating cache directory %s: %s"
                                 % (self.cachedir, str(e)))
        if not os.access(self.cachedir, os.W_OK):
            raise CacheError("Cache directory %s not writable" % self.cachedir)


class CacheError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


# A global cache instance, set by the appropriate test runner.
conversion_cache = None
