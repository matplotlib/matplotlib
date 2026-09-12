import json
from pathlib import Path
import shutil
import sys
from unittest import mock

from matplotlib import cbook, dviread as dr
from matplotlib.testing import subprocess_run_for_testing, _has_tex_package
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


@pytest.mark.skipif(sys.platform == "emscripten" or shutil.which("kpsewhich") is None,
                    reason="kpsewhich is not available")
@pytest.mark.parametrize("engine", ["pdflatex", "xelatex", "lualatex"])
def test_dviread(tmp_path, engine, monkeypatch):
    dirpath = Path(__file__).parent / "baseline_images/dviread"
    shutil.copy(dirpath / "test.tex", tmp_path)
    shutil.copy(cbook._get_data_path("fonts/ttf/DejaVuSans.ttf"), tmp_path)
    cmd, fmt = {
        "pdflatex": (["latex", "-no-shell-escape"], "dvi"),
        "xelatex": (["xelatex", "-no-pdf", "-no-shell-escape"], "xdv"),
        "lualatex": (["lualatex", "-output-format=dvi", "-no-shell-escape"], "dvi"),
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
        ["latex", "-no-shell-escape", "test.tex"],
        cwd=tmp_path, check=True, capture_output=True)
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


# About this test: _LuatexKpsewhich.search() communicates with a long-lived `luatex`
# subprocess over pipes and turns whatever comes back into one of three
# outcomes that its caller (find_tex_file) depends on:
#   * a path string  -> luatex found the file;
#   * ""             -> luatex works but the file genuinely doesn't exist;
#   * None           -> luatex is unusable, so the caller must fall back to
#                       running kpsewhich directly.
# Each parametrized case below feeds search() one scenario and asserts it maps
# to the right outcome.
#
# Notice: We use mock-based testing here for cross-platform compatibility and
# deterministic/standardized testing. This means that testing is quite heavily
# tied to the internals of the tested function. Restructuring or rewriting of
# code might break testing without actually changing the overall behaviour.
# This tradeoff was accepted in #31984.
@pytest.mark.parametrize("readline, write_error, expected", [
    # (bytes luatex "prints", exception raised on write, expected return)
    # Case 1: supply a path with no error; check search() passes it through
    (b"/texmf/pdftex.map\n", None, "/texmf/pdftex.map"),
    # Case 2: make luatex answer "nil" (file missing); check search() returns ""
    (b"nil\n", None, ""),
    # Case 3: supply an empty read (luatex died); check search() returns None
    (b"", None, None),
    # Case 4: raise an error on write (pipe broke); check search() returns None
    (b"unused", BrokenPipeError, None),
])
def test_luatexkpsewhich_search(readline, write_error, expected):
    proc = mock.Mock()  # stand-in for the luatex subprocess
    proc.poll.return_value = None  # so search() won't mark it dead and restart it
    proc.stdin.write.side_effect = write_error  # None => write ok; else pipe broke
    proc.stdout.readline.return_value = readline  # luatex's answer, read by search()
    lk = object.__new__(dr._LuatexKpsewhich)  # bypass __new__ (spawns a real luatex)
    lk._proc = proc  # hand search() our fake process instead
    assert lk.search("pdftex.map") == expected


def test_find_tex_file_fallback_to_kpsewhich(monkeypatch):
    # Unlike the search() test above, this checks the routing in find_tex_file,
    # which takes the result of search() and acts accordingly by either calling
    # the fallback (kpsewhich) if the search failed via luatex, or just forwarding
    # the results. We therefore replace the whole _LuatexKpsewhich with a stub
    # whose search() always returns None (luatex unusable), and check that
    # find_tex_file then falls back to kpsewhich.
    class _UnusableLuatex:
        def search(self, filename):
            return None

    monkeypatch.setattr(dr, "_LuatexKpsewhich", _UnusableLuatex)

    # kpsewhich is run via cbook._check_and_log_subprocess; intercept it so no
    # real process is spawned, record what was called, and hand back a path.
    commands = []

    def fake_check_and_log_subprocess(command, logger, **kwargs):
        commands.append(command)
        return "/texmf/pdftex.map\n"

    monkeypatch.setattr(
        cbook, "_check_and_log_subprocess", fake_check_and_log_subprocess)

    # cache_clear() before and after: find_tex_file is lru_cached, so a stale
    # entry could hide the behaviour we test / leak our injected fake function
    # into other tests.
    dr.find_tex_file.cache_clear()
    try:
        assert dr.find_tex_file("pdftex.map") == "/texmf/pdftex.map"
    finally:
        dr.find_tex_file.cache_clear()
    # Check if the intercepted command was indeed kpsewhich (fallback was called).
    assert commands and commands[0][0] == "kpsewhich"


def test_find_tex_file_no_fallback_when_luatex_reports_missing(monkeypatch):
    # This exactly mirrors the one above.  Here,
    # search() returns "" (luatex works, file genuinely missing), which must
    # explicitly not trigger the kpsewhich fallback. find_tex_file should just
    # raise.
    class _WorkingLuatex:
        def search(self, filename):
            return ""

    monkeypatch.setattr(dr, "_LuatexKpsewhich", _WorkingLuatex)

    # Guard: if the fallback wrongly fired, kpsewhich would run and fail loudly.
    def fail_if_called(command, logger, **kwargs):
        raise AssertionError("kpsewhich should not be called")

    monkeypatch.setattr(cbook, "_check_and_log_subprocess", fail_if_called)

    # cache_clear() around the call, as above, to isolate this from other tests.
    dr.find_tex_file.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            dr.find_tex_file("missing.tfm")
    finally:
        dr.find_tex_file.cache_clear()
