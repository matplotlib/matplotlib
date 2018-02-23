from matplotlib.testing.decorators import skip_if_command_unavailable

try:
    from unittest import mock
except ImportError:
    import mock

import matplotlib.dviread as dr
import os.path
import json
import pytest
import sqlite3
import warnings


def test_PsfontsMap(monkeypatch):
    monkeypatch.setattr(dr, 'find_tex_file', lambda x: x)

    filename = os.path.join(
        os.path.dirname(__file__),
        'baseline_images', 'dviread', 'test.map')
    fontmap = dr.PsfontsMap(filename)
    # Check all properties of a few fonts
    for n in [1, 2, 3, 4, 5]:
        key = ('TeXfont%d' % n).encode('ascii')
        entry = fontmap[key]
        assert entry.texname == key
        assert entry.psname == ('PSfont%d' % n).encode('ascii')
        if n not in [3, 5]:
            assert entry.encoding == ('font%d.enc' % n).encode('ascii')
        elif n == 3:
            assert entry.encoding == b'enc3.foo'
        # We don't care about the encoding of TeXfont5, which specifies
        # multiple encodings.
        if n not in [1, 5]:
            assert entry.filename == ('font%d.pfa' % n).encode('ascii')
        else:
            assert entry.filename == ('font%d.pfb' % n).encode('ascii')
        if n == 4:
            assert entry.effects == {'slant': -0.1, 'extend': 2.2}
        else:
            assert entry.effects == {}
    # Some special cases
    entry = fontmap[b'TeXfont6']
    assert entry.filename is None
    assert entry.encoding is None
    entry = fontmap[b'TeXfont7']
    assert entry.filename is None
    assert entry.encoding == b'font7.enc'
    entry = fontmap[b'TeXfont8']
    assert entry.filename == b'font8.pfb'
    assert entry.encoding is None
    entry = fontmap[b'TeXfont9']
    assert entry.filename == b'/absolute/font9.pfb'
    # Missing font
    with pytest.raises(KeyError) as exc:
        fontmap[b'no-such-font']
    assert 'no-such-font' in str(exc.value)


@skip_if_command_unavailable(["kpsewhich", "-version"])
def test_dviread():
    dir = os.path.join(os.path.dirname(__file__), 'baseline_images', 'dviread')
    with open(os.path.join(dir, 'test.json')) as f:
        correct = json.load(f)
        for entry in correct:
            entry['text'] = [[a, b, c, d.encode('ascii'), e]
                             for [a, b, c, d, e] in entry['text']]
    with dr.Dvi(os.path.join(dir, 'test.dvi'), None) as dvi:
        data = [{'text': [[t.x, t.y,
                           chr(t.glyph),
                           t.font.texname,
                           round(t.font.size, 2)]
                          for t in page.text],
                 'boxes': [[b.x, b.y, b.height, b.width] for b in page.boxes]}
                for page in dvi]
    assert data == correct


@skip_if_command_unavailable(["kpsewhich", "-version"])
def test_dviread_get_fonts():
    dir = os.path.join(os.path.dirname(__file__), 'baseline_images', 'dviread')
    dvi = dr._DviReader(os.path.join(dir, 'test.dvi'), None)
    assert dvi.fontnames == \
        {'cmex10', 'cmmi10', 'cmmi5', 'cmr10', 'cmr5', 'cmr7'}
    vf = dr.Vf(os.path.join(dir, 'virtual.vf'))
    assert vf.fontnames == {'cmex10', 'cmr10'}


def test_dviread_get_fonts_error_handling():
    dir = os.path.join(os.path.dirname(__file__), 'baseline_images', 'dviread')
    for n, message in [(1, "too few 223 bytes"),
                       (2, "post-postamble identification"),
                       (3, "postamble offset"),
                       (4, "postamble not found"),
                       (5, "opcode 127 in postamble")]:
        with pytest.raises(ValueError) as e:
            dr.Dvi(os.path.join(dir, "broken%d.dvi" % n), None)
        assert message in str(e.value)


def test_TeXSupportCache(tmpdir):
    dbfile = str(tmpdir / "test.db")
    cache = dr.TeXSupportCache(filename=dbfile)
    assert cache.get_pathnames(['foo', 'bar']) == {}
    with cache.connection as transaction:
        cache.update_pathnames({'foo': '/tmp/foo',
                                'xyzzy': '/xyzzy.dat',
                                'fontfile': None}, transaction)
    assert cache.get_pathnames(['foo', 'bar']) == {'foo': '/tmp/foo'}
    assert cache.get_pathnames(['xyzzy', 'fontfile']) == \
        {'xyzzy': '/xyzzy.dat', 'fontfile': None}

    # check that modifying a dvi file invalidates the cache
    filename = str(tmpdir / "file.dvi")
    with open(filename, "wb") as f:
        f.write(b'qwerty')
    os.utime(filename, (0, 0))
    with cache.connection as t:
        id1 = cache.dvi_new_file(filename, t)
    assert cache.dvi_id(filename) == id1

    with open(filename, "wb") as f:
        f.write(b'asfdg')
    os.utime(filename, (0, 0))
    assert cache.dvi_id(filename) is None
    with cache.connection as t:
        id2 = cache.dvi_new_file(filename, t)
    assert cache.dvi_id(filename) == id2


def test_TeXSupportCache_versioning(tmpdir):
    dbfile = str(tmpdir / "test.db")
    cache1 = dr.TeXSupportCache(dbfile)
    with cache1.connection as transaction:
        cache1.update_pathnames({'foo': '/tmp/foo'}, transaction)

    with sqlite3.connect(dbfile, isolation_level="DEFERRED") as conn:
        conn.executescript('PRAGMA user_version=1000000000;')

    with pytest.raises(dr.TeXSupportCacheError):
        cache2 = dr.TeXSupportCache(dbfile)


def test_find_tex_files(tmpdir):
    with mock.patch('matplotlib.dviread.subprocess.Popen') as mock_popen:
        mock_proc = mock.Mock()
        stdout = '{s}tmp{s}foo.pfb\n{s}tmp{s}bar.map\n'.\
                 format(s=os.path.sep).encode('ascii')
        mock_proc.configure_mock(**{'communicate.return_value': (stdout, b'')})
        mock_popen.return_value = mock_proc

        # first call uses the results from kpsewhich
        cache = dr.TeXSupportCache(filename=str(tmpdir / "test.db"))
        assert dr.find_tex_files(
            ['foo.pfb', 'cmsy10.pfb', 'bar.tmp', 'bar.map'], cache) \
            == {'foo.pfb': '{s}tmp{s}foo.pfb'.format(s=os.path.sep),
                'bar.map': '{s}tmp{s}bar.map'.format(s=os.path.sep),
                'cmsy10.pfb': None, 'bar.tmp': None}
        assert mock_popen.called

        # second call (subset of the first one) uses only the cache
        mock_popen.reset_mock()
        assert dr.find_tex_files(['foo.pfb', 'cmsy10.pfb'], cache) \
            == {'foo.pfb': '{s}tmp{s}foo.pfb'.format(s=os.path.sep),
                'cmsy10.pfb': None}
        assert not mock_popen.called

        # third call (includes more than the first one) uses kpsewhich again
        mock_popen.reset_mock()
        stdout = '{s}usr{s}local{s}cmr10.tfm\n'.\
                 format(s=os.path.sep).encode('ascii')
        mock_proc.configure_mock(**{'communicate.return_value': (stdout, b'')})
        mock_popen.return_value = mock_proc
        assert dr.find_tex_files(['foo.pfb', 'cmr10.tfm'], cache) == \
            {'foo.pfb': '{s}tmp{s}foo.pfb'.format(s=os.path.sep),
             'cmr10.tfm': '{s}usr{s}local{s}cmr10.tfm'.format(s=os.path.sep)}
        assert mock_popen.called


def test_find_tex_file_format():
    with mock.patch('matplotlib.dviread.subprocess.Popen') as mock_popen:
        mock_proc = mock.Mock()
        stdout = b'/foo/bar/baz\n'
        mock_proc.configure_mock(**{'communicate.return_value': (stdout, b'')})
        mock_popen.return_value = mock_proc

        warnings.filterwarnings(
            'ignore',
            'The format option to find_tex_file is deprecated.*',
            UserWarning)
        assert dr.find_tex_file('foobar', format='tfm') == '/foo/bar/baz'
        assert mock_popen.called
