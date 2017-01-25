from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
try:
    from unittest import mock
except ImportError:
    import mock

import os
import shutil
import stat
import tempfile

from nose.tools import raises

from matplotlib import cbook
from matplotlib.testing.conversion_cache import ConversionCache, CacheError


def test_cache_basic():
    tmpdir = tempfile.mkdtemp()

    def intmp(f):
        return os.path.join(tmpdir, f)
    try:
        cache = ConversionCache(intmp('cache'))
        with open(intmp('fake.pdf'), 'w') as pdf:
            pdf.write('this is a fake pdf file')
        with open(intmp('fake.svg'), 'w') as svg:
            svg.write('this pretends to be an svg file')

        assert not cache.get(intmp('fake.pdf'), intmp('fakepdf.png'))
        assert not cache.get(intmp('fake.svg'), intmp('fakesvg.png'))
        assert cache.report() == \
            {'gets': {intmp('fake.pdf'), intmp('fake.svg')},
             'hits': set()}

        with open(intmp('fakepdf.png'), 'w') as png:
            png.write('generated from the pdf file')
        cache.put(intmp('fake.pdf'), intmp('fakepdf.png'))
        assert cache.get(intmp('fake.pdf'), intmp('copypdf.png'))
        with open(intmp('copypdf.png'), 'r') as copy:
            assert copy.read() == 'generated from the pdf file'
        assert cache.report() == \
            {'gets': {intmp('fake.pdf'), intmp('fake.svg')},
             'hits': set([intmp('fake.pdf')])}

        with open(intmp('fakesvg.png'), 'w') as png:
            png.write('generated from the svg file')
        cache.put(intmp('fake.svg'), intmp('fakesvg.png'))
        assert cache.get(intmp('fake.svg'), intmp('copysvg.png'))
        with open(intmp('copysvg.png'), 'r') as copy:
            assert copy.read() == 'generated from the svg file'
        assert cache.report() == \
            {'gets': {intmp('fake.pdf'), intmp('fake.svg')},
             'hits': {intmp('fake.pdf'), intmp('fake.svg')}}
    finally:
        shutil.rmtree(tmpdir)


def test_cache_expire():
    tmpdir = tempfile.mkdtemp()

    def intmp(*f):
        return os.path.join(tmpdir, *f)
    try:
        cache = ConversionCache(intmp('cache'), 10)
        for i in range(5):
            filename = intmp('cache', 'file%d.png' % i)
            with open(filename, 'w') as f:
                f.write('1234')
            os.utime(filename, (i*1000, i*1000))

        cache.expire()
        assert not os.path.exists(intmp('cache', 'file0.png'))
        assert not os.path.exists(intmp('cache', 'file1.png'))
        assert not os.path.exists(intmp('cache', 'file2.png'))
        assert os.path.exists(intmp('cache', 'file3.png'))
        assert os.path.exists(intmp('cache', 'file4.png'))

        with open(intmp('cache', 'onemore.png'), 'w') as f:
            f.write('x' * 11)
        os.utime(intmp('cache', 'onemore.png'), (5000, 5000))

        cache.expire()
        assert not os.path.exists(intmp('cache', 'file0.png'))
        assert not os.path.exists(intmp('cache', 'file1.png'))
        assert not os.path.exists(intmp('cache', 'file2.png'))
        assert not os.path.exists(intmp('cache', 'file3.png'))
        assert not os.path.exists(intmp('cache', 'file4.png'))
        assert not os.path.exists(intmp('cache', 'onemore.png'))

    finally:
        shutil.rmtree(tmpdir)


def test_cache_default_dir():
    try:
        path = ConversionCache.get_cache_dir()
        assert path.endswith('test_cache')
    except CacheError:
        pass


@raises(_CacheError)
@mock.patch('matplotlib.testing._conversion_cache.cbook.mkdirs',
            side_effect=IOError)
def test_cache_mkdir_error(mkdirs):
    tmpdir = tempfile.mkdtemp()
    try:
        c = ConversionCache(os.path.join(tmpdir, 'cache'))
    finally:
        shutil.rmtree(tmpdir)


@raises(_CacheError)
@mock.patch('matplotlib.testing._conversion_cache.os.access',
            side_effect=[False])
def test_cache_unwritable_error(access):
    tmpdir = tempfile.mkdtemp()
    cachedir = os.path.join(tmpdir, 'test_cache')
    try:
        cbook.mkdirs(cachedir)
        c = ConversionCache(cachedir)
    finally:
        shutil.rmtree(tmpdir)
