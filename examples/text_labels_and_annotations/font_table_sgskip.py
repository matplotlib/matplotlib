"""
==========
Font table
==========

Matplotlib's font support is provided by the FreeType library.

Here, we use `~.Axes.table` build a font table that shows the glyphs by Unicode
codepoint, and print the glyphs corresponding to codepoints beyond 0xff.

Download this script and run it to investigate a font by running ::

    python font_table_sgskip.py /path/to/font/file
"""

import unicodedata

import matplotlib.font_manager as fm
from matplotlib.ft2font import FT2Font
import matplotlib.pyplot as plt
import numpy as np


def draw_font_table(path):
    """
    Parameters
    ----------
    path : str or None
        The path to the font file.  If None, use Matplotlib's default font.
    """

    if path is None:
        path = fm.findfont(fm.FontProperties())  # The default font.

    font = FT2Font(path)
    # A charmap is a mapping of "character codes" (in the sense of a character
    # encoding, e.g. latin-1) to glyph indices (i.e. the internal storage table
    # of the font face).
    # In FreeType>=2.1, a Unicode charmap (i.e. mapping Unicode codepoints)
    # is selected by default.  Moreover, recent versions of FreeType will
    # automatically synthesize such a charmap if the font does not include one
    # (this behavior depends on the font format; for example it is present
    # since FreeType 2.0 for Type 1 fonts but only since FreeType 2.8 for
    # TrueType (actually, SFNT) fonts).
    # The code below (specifically, the ``chr(char_code)`` call) assumes that
    # we have indeed selected a Unicode charmap.
    codes = font.get_charmap().items()

    labelc = ["{:X}".format(i) for i in range(16)]
    labelr = ["{:02X}".format(16 * i) for i in range(16)]
    chars = [["" for c in range(16)] for r in range(16)]
    non_8bit = []

    for char_code, glyph_index in codes:
        char = chr(char_code)
        if char_code >= 256:
            non_8bit.append((
                str(glyph_index),
                char,
                unicodedata.name(
                    char,
                    f"{char_code:#x} ({font.get_glyph_name(glyph_index)})"),
            ))
            continue
        r, c = divmod(char_code, 16)
        chars[r][c] = char
    if non_8bit:
        indices, *_ = zip(*non_8bit)
        max_indices_len = max(map(len, indices))
        print("The font face contains the following glyphs corresponding to "
              "code points beyond 0xff:")
        for index, char, name in non_8bit:
            print(f"{index:>{max_indices_len}} {char} {name}")

    ax = plt.figure(figsize=(8, 4), dpi=120).subplots()
    ax.set_title(path)
    ax.set_axis_off()

    table = ax.table(
        cellText=chars,
        rowLabels=labelr,
        colLabels=labelc,
        rowColours=["palegreen"] * 16,
        colColours=["palegreen"] * 16,
        cellColours=[[".95" for c in range(16)] for r in range(16)],
        cellLoc='center',
        loc='upper left',
    )
    for key, cell in table.get_celld().items():
        row, col = key
        if row > 0 and col > -1:  # Beware of table's idiosyncratic indexing...
            cell.set_text_props(fontproperties=fm.FontProperties(fname=path))

    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("path", nargs="?", help="Path to the font file.")
    args = parser.parse_args()

    draw_font_table(args.path)
