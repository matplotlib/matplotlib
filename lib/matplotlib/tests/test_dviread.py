import json
from pathlib import Path
import shutil

from matplotlib import cbook, dviread as dr
from matplotlib.testing import subprocess_run_for_testing, _has_tex_package
from matplotlib.texmanager import TexManager
import pytest

def test_ops():
    filename = str(Path(__file__).parent / 'baseline_images/dviread/color.dvi')
    Op = dr.Ops.Op
    set_chars = lambda s: [Op(ord(c), 'set_char', {'c': ord(c)}) for c in s]
    assert list(dr.Ops.read_file(filename)) == [
        Op(247, 'pre', {
            'i': 2, 'num': 25400000, 'den': 473628672, 'mag': 1000, 'k': 27,
            'cmnt': b' TeX output 2026.03.10:0955'}),
        Op(139, 'bop', {
            'c0': 1, 'c1': 0, 'c2': 0, 'c3': 0, 'c4': 0, 'c5': 0,
            'c6': 0, 'c7': 0, 'c8': 0, 'c9': 0, 'p': -1}),
        Op(141, 'push', {}),
        Op(239, 'special', {'k': 26, 'text': b'header=l3backend-dvips.pro'}),
        Op(239, 'special', {'k': 35, 'text': b'papersize=5203.43999pt,5203.43999pt'}),
        Op(239, 'special', {'k': 35, 'text': b'papersize=5203.43999pt,5203.43999pt'}),
        Op(142, 'pop', {}),
        Op(160, 'down', {'amount': 333506151}),
        Op(141, 'push', {}),
        Op(160, 'down', {'amount': -335144551}),
        Op(141, 'push', {}),
        Op(141, 'push', {}),
        Op(239, 'special', {'k': 17, 'text': b'color push  Black'}),
        Op(146, 'right', {'amount': 331540071}),
        Op(239, 'special', {'k': 9, 'text': b'color pop'}),
        Op(142, 'pop', {}),
        Op(142, 'pop', {}),
        Op(160, 'down', {'amount': 333178471}),
        Op(141, 'push', {}),
        Op(160, 'down', {'amount': -330098279}),
        Op(141, 'push', {}),
        Op(145, 'right', {'amount': 983040}),
        Op(243, 'fnt_def', {
            'k': 28, 'c': 2194559542, 's': 786432, 'd': 786432, 'a': 0, 'l': 6,
            'area': b'', 'name': b'cmss12'}),
        Op(199, 'fnt_num', {'n': 28}),
    ] + set_chars("Default,") + [
        Op(145, 'right', {'amount': 475130}),
        Op(239, 'special', {'k': 28, 'text': b'color push rgb 1.0  0.0  0.0'}),
    ] + set_chars("red") + [
        Op(144, 'right', {'amount': 20984}),
        Op(141, 'push', {}),
        Op(141, 'push', {}),
        Op(159, 'down', {'amount': -174762}),
        Op(132, 'set_rule', {'height': 65536, 'width': 65536}),
        Op(142, 'pop', {}),
        Op(142, 'pop', {}),
        Op(150, 'w', args={'new_w': 65536}),
        Op(141, 'push', {}),
        Op(141, 'push', {}),
        Op(159, 'down', {'amount': -174762}),
        Op(132, 'set_rule', {'height': 65536, 'width': 65536}),
        Op(142, 'pop', {}),
        Op(142, 'pop', {}),
    ] + ([
        # The red line is apparently a bunch of little red lines.
        Op(147, 'w0', {}),
        Op(141, 'push', {}),
        Op(141, 'push', {}),
        Op(159, 'down', {'amount': -174762}),
        Op(132, 'set_rule', {'height': 65536, 'width': 65536}),
        Op(142, 'pop', {}),
        Op(142, 'pop', {}),
    ] * 83) + [
        Op(145, 'right', {'amount': 68031}),
        Op(239, 'special', {'k': 9, 'text': b'color pop'}),
    ] + set_chars(",and") + [
        Op(150, 'w', args={'new_w': 256680}),
    ] + set_chars("back") + [
        Op(147, 'w0', args={}),
    ] + set_chars("again.") + [
        Op(142, 'pop', {}),
        Op(142, 'pop', {}),
        Op(159, 'down', {'amount': 1966080}),
        Op(141, 'push', {}),
        Op(239, 'special', {'k': 17, 'text': b'color push  Black'}),
        Op(146, 'right', {'amount': 331540071}),
        Op(239, 'special', {'k': 9, 'text': b'color pop'}),
        Op(142, 'pop', {}),
        Op(142, 'pop', {}),
        Op(140, 'eop', {}),
        Op(248, 'post', {
            'p': 42, 'num': 25400000, 'den': 473628672, 'mag': 1000,
            'l': 333506151, 'u': 331540071, 's': 5, 't': 1}),
        Op(243, 'fnt_def', {
            'k': 28, 'c': 2194559542, 's': 786432, 'd': 786432, 'a': 0, 'l': 6,
            'area': b'', 'name': b'cmss12'}),
        Op(249, 'post_post', {'q': 1939, 'i': 2, 'padding': 3755991007}),
    ]

def test_PsfontsMap(monkeypatch):
    monkeypatch.setattr(dr, 'find_tex_file', lambda x: x.decode())

    filename = str(Path(__file__).parent / 'baseline_images/dviread/test.map')
    fontmap = dr.PsfontsMap(filename)
    # Check all properties of a few fonts
    for n in [1, 2, 3, 4, 5]:
        key = b'TeXfont%d' % n
        entry = fontmap[key]
        assert entry.texname == key
        assert entry.psname == b'PSfont%d' % n
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
            assert entry.effects == {'slant': -0.1, 'extend': 1.2}
        else:
            assert entry.effects == {}
    # Some special cases
    entry = fontmap[b'TeXfont6']
    assert entry.filename is None
    assert entry.encoding is None
    entry = fontmap[b'TeXfont7']
    assert entry.filename is None
    assert entry.encoding == 'font7.enc'
    entry = fontmap[b'TeXfont8']
    assert entry.filename == 'font8.pfb'
    assert entry.encoding is None
    entry = fontmap[b'TeXfont9']
    assert entry.psname == b'TeXfont9'
    assert entry.filename == '/absolute/font9.pfb'
    # First of duplicates only.
    entry = fontmap[b'TeXfontA']
    assert entry.psname == b'PSfontA1'
    # Slant/Extend only works for T1 fonts.
    entry = fontmap[b'TeXfontB']
    assert entry.psname == b'PSfontB6'
    # Subsetted TrueType must have encoding.
    entry = fontmap[b'TeXfontC']
    assert entry.psname == b'PSfontC3'
    # Missing font
    with pytest.raises(LookupError, match='no-such-font'):
        fontmap[b'no-such-font']
    with pytest.raises(LookupError, match='%'):
        fontmap[b'%']


@pytest.mark.skipif(shutil.which("kpsewhich") is None,
                    reason="kpsewhich is not available")
@pytest.mark.parametrize("engine", ["pdflatex", "xelatex", "lualatex"])
def test_dviread(tmp_path, engine, monkeypatch):
    dirpath = Path(__file__).parent / "baseline_images/dviread"
    shutil.copy(dirpath / "test.tex", tmp_path)
    shutil.copy(cbook._get_data_path("fonts/ttf/DejaVuSans.ttf"), tmp_path)
    cmd, fmt = {
        "pdflatex": (["latex"], "dvi"),
        "xelatex": (["xelatex", "-no-pdf"], "xdv"),
        "lualatex": (["lualatex", "-output-format=dvi"], "dvi"),
    }[engine]
    if shutil.which(cmd[0]) is None:
        pytest.skip(f"{cmd[0]} is not available")
    subprocess_run_for_testing(
        [*cmd, "test.tex"], cwd=tmp_path, check=True, capture_output=True)
    # dviread must be run from the tmppath directory because {xe,lua}tex output
    # records the path to DejaVuSans.ttf as it is written in the tex source,
    # i.e. as a relative path.
    monkeypatch.chdir(tmp_path)
    with dr.Dvi(tmp_path / f"test.{fmt}", None) as dvi:
        try:
            pages = [*dvi]
        except FileNotFoundError as exc:
            for note in getattr(exc, "__notes__", []):
                if "too-old version of luaotfload" in note:
                    pytest.skip(note)
            raise
    data = [
        {
            "text": [
                [
                    t.x, t.y,
                    t._as_unicode_or_name(),
                    t.font.resolve_path().name,
                    round(t.font.size, 2),
                    t.font.effects,
                ] for t in page.text
            ],
            "boxes": [[b.x, b.y, b.height, b.width] for b in page.boxes]
        } for page in pages
    ]
    correct = json.loads((dirpath / f"{engine}.json").read_text())
    assert data == correct


@pytest.mark.skipif(shutil.which("latex") is None, reason="latex is not available")
@pytest.mark.skipif(not _has_tex_package("concmath"), reason="needs concmath.sty")
def test_dviread_pk(tmp_path):
    (tmp_path / "test.tex").write_text(r"""
        \documentclass{article}
        \usepackage{concmath}
        \pagestyle{empty}
        \begin{document}
        Hi!
        \end{document}
        """)
    subprocess_run_for_testing(
        ["latex", "test.tex"], cwd=tmp_path, check=True, capture_output=True)
    with dr.Dvi(tmp_path / "test.dvi", None) as dvi:
        pages = [*dvi]
    data = [
        {
            "text": [
                [
                    t.x, t.y,
                    t._as_unicode_or_name(),
                    t.font.resolve_path().name,
                    round(t.font.size, 2),
                    t.font.effects,
                ] for t in page.text
            ],
            "boxes": [[b.x, b.y, b.height, b.width] for b in page.boxes]
        } for page in pages
    ]
    correct = [{
        'boxes': [],
        'text': [
            [5046272, 4128768, 'H?', 'ccr10.600pk', 9.96, {}],
            [5530510, 4128768, 'i?', 'ccr10.600pk', 9.96, {}],
            [5716195, 4128768, '!?', 'ccr10.600pk', 9.96, {}],
        ],
    }]
    assert data == correct
