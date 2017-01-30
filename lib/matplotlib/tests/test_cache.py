from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
try:
    from unittest import mock
except ImportError:
    import mock

import os

from matplotlib import cbook
from matplotlib.testing._conversion_cache import _ConversionCache, _CacheError
import pytest


def test_cache_basic(tmpdir):
    def intmp(f):
        return tmpdir.join(f)

    def fname(f):
        return str(intmp(f))

    try:
        cache = _ConversionCache(fname('cache'))
        intmp('fake.pdf').write_binary(b'this is a fake pdf file')
        intmp('fake.svg').write_binary(b'this pretends to be an svg file')
        assert not cache.get(fname('fake.pdf'), fname('fakepdf.png'))
        assert not cache.get(fname('fake.svg'), fname('fakesvg.png'))
        assert cache.report() == \
            {'gets': {fname('fake.pdf'), fname('fake.svg')},
             'hits': set()}

        intmp('fakepdf.png').write_binary(b'generated from the pdf file')
        cache.put(fname('fake.pdf'), fname('fakepdf.png'))
        assert cache.get(fname('fake.pdf'), fname('copypdf.png'))
        assert intmp('copypdf.png').read() == 'generated from the pdf file'
        assert cache.report() == \
            {'gets': {fname('fake.pdf'), fname('fake.svg')},
             'hits': {fname('fake.pdf')}}

        intmp('fakesvg.png').write_binary(b'generated from the svg file')
        cache.put(fname('fake.svg'), fname('fakesvg.png'))
        assert cache.get(fname('fake.svg'), fname('copysvg.png'))
        assert intmp('copysvg.png').read() == 'generated from the svg file'
        assert cache.report() == \
            {'gets': {fname('fake.pdf'), fname('fake.svg')},
             'hits': {fname('fake.pdf'), fname('fake.svg')}}
    finally:
        tmpdir.remove(rec=1)


def test_cache_expire(tmpdir):
    def intmp(*f):
        return tmpdir.join(*f)

    def fname(*f):
        return str(intmp(*f))

    try:
        cache = _ConversionCache(fname('cache'), 10)
        for i in range(5):
            pngfile = intmp('cache', 'file%d.png' % i)
            pngfile.write_binary(b'1234')
            os.utime(str(pngfile), (i*1000, i*1000))

        cache.expire()
        assert not os.path.exists(fname('cache', 'file0.png'))
        assert not os.path.exists(fname('cache', 'file1.png'))
        assert not os.path.exists(fname('cache', 'file2.png'))
        assert os.path.exists(fname('cache', 'file3.png'))
        assert os.path.exists(fname('cache', 'file4.png'))

        intmp('cache', 'onemore.png').write_binary(b'x' * 11)
        os.utime(fname('cache', 'onemore.png'), (5000, 5000))

        cache.expire()
        assert not os.path.exists(fname('cache', 'file0.png'))
        assert not os.path.exists(fname('cache', 'file1.png'))
        assert not os.path.exists(fname('cache', 'file2.png'))
        assert not os.path.exists(fname('cache', 'file3.png'))
        assert not os.path.exists(fname('cache', 'file4.png'))
        assert not os.path.exists(fname('cache', 'onemore.png'))

    finally:
        tmpdir.remove(rec=1)


def test_cache_default_dir():
    try:
        path = _ConversionCache.get_cache_dir()
        assert path.endswith('test_cache')
    except _CacheError:
        pass


@mock.patch('matplotlib.testing._conversion_cache.cbook.mkdirs',
            side_effect=OSError)
def test_cache_mkdir_error(mkdirs, tmpdir):
    with pytest.raises(_CacheError):
        try:
            c = _ConversionCache(str(tmpdir.join('cache')))
        finally:
            tmpdir.remove(rec=1)


@mock.patch('matplotlib.testing._conversion_cache.os.access',
            side_effect=[False])
def test_cache_unwritable_error(access, tmpdir):
    with pytest.raises(_CacheError):
        cachedir = tmpdir.join('cache')
        cachedir.ensure(dir=True)
        try:
            c = _ConversionCache(str(cachedir))
        finally:
            tmpdir.remove(rec=1)
