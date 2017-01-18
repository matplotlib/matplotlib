from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from matplotlib.testing.decorators import skip_if_command_unavailable

import matplotlib.dviread as dr
import os.path
import json


def test_PsfontsMap(monkeypatch):
    monkeypatch.setattr(dr, 'find_tex_file', lambda x: x)

    filename = os.path.join(
        os.path.dirname(__file__),
        'baseline_images', 'dviread', 'test.map')
    fontmap = dr.PsfontsMap(filename)
    # Check all properties of a few fonts
    for n in [1, 2, 3, 4, 5]:
        key = 'TeXfont%d' % n
        entry = fontmap[key]
        assert entry.texname == key
        assert entry.psname == 'PSfont%d' % n
        if n not in [3, 5]:
            assert entry.encoding == 'font%d.enc' % n
        elif n == 3:
            assert entry.encoding == 'enc3.foo'
        # We don't care about the encoding of TeXfont5, which specifies
        # multiple encodings.
        if n not in [1, 5]:
            assert entry.filename == 'font%d.pfa' % n
        else:
            assert entry.filename == 'font%d.pfb' % n
        if n == 4:
            assert entry.effects == {'slant': -0.1, 'extend': 2.2}
        else:
            assert entry.effects == {}
    # Some special cases
    entry = fontmap['TeXfont6']
    assert entry.filename is None
    assert entry.encoding is None
    entry = fontmap['TeXfont7']
    assert entry.filename is None
    assert entry.encoding == 'font7.enc'
    entry = fontmap['TeXfont8']
    assert entry.filename == 'font8.pfb'
    assert entry.encoding is None
    entry = fontmap['TeXfont9']
    assert entry.filename == '/absolute/font9.pfb'


@skip_if_command_unavailable(["kpsewhich", "-version"])
def test_dviread():
    dir = os.path.join(os.path.dirname(__file__), 'baseline_images', 'dviread')
    with open(os.path.join(dir, 'test.json')) as f:
        correct = json.load(f)
    with dr.Dvi(os.path.join(dir, 'test.dvi'), None) as dvi:
        data = [{'text': [[t.x, t.y,
                           six.unichr(t.glyph),
                           six.text_type(t.font.texname),
                           round(t.font.size, 2)]
                          for t in page.text],
                 'boxes': [[b.x, b.y, b.height, b.width] for b in page.boxes]}
                for page in dvi]
    assert data == correct
