import json
from pathlib import Path
import shutil
import subprocess

import matplotlib.dviread as dr
import pytest


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
def test_dviread(tmp_path, engine):
    dirpath = Path(__file__).parent / "baseline_images/dviread"
    shutil.copy(dirpath / "test.tex", tmp_path)
    if engine == "pdflatex":
        if shutil.which("latex") is None:
            pytest.skip("latex is not available")
        cmd = ["latex"]
        fmt = "dvi"
    elif engine == "xelatex":
        if shutil.which("xelatex") is None:
            pytest.skip("xelatex is not available")
        cmd = ["xelatex", "-no-pdf"]
        fmt = "xdv"
    elif engine == "lualatex":
        if shutil.which("lualatex") is None:
            pytest.skip("lualatex is not available")
        cmd = ["lualatex", "-output-format=dvi"]
        fmt = "dvi"
    subprocess.run(
        [*cmd, "test.tex"], cwd=tmp_path, check=True, capture_output=True)
    with dr.Dvi(tmp_path / f"test.{fmt}", None) as dvi:
        data = [
            {
                "text": [
                    [
                        t.x, t.y,
                        t._as_unicode_or_name(),
                        fontname
                        if (fontname := t.font.texname.decode())[0] != "["
                        else Path(fontname[1:-1]).name,
                        round(t.font.size, 2),
                    ] for t in page.text
                ],
                "boxes": [[b.x, b.y, b.height, b.width] for b in page.boxes]
            } for page in dvi
        ]
    correct = json.loads((dirpath / f"{engine}.json").read_text())
    assert data == correct
