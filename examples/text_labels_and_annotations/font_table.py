"""
==========
Font table
==========

Matplotlib's font support is provided by the FreeType library.

Here, we use `~.Axes.table` to draw a table that shows the glyphs by Unicode
codepoint. For brevity, the table only contains the first 256 glyphs.

The example is a full working script. You can download it and use it to
investigate a font by running ::

    python font_table.py /path/to/font/file
"""

import unicodedata

import matplotlib.font_manager as fm
from matplotlib.ft2font import FT2Font
import matplotlib.pyplot as plt


def draw_font_table(path, print_non_latin1=False):
    """
    Parameters
    ----------
    path : str or None
        The path to the font file.  If None, use Matplotlib's default font.
    print_non_latin1 : bool
        Print non-latin1 chars (i.e. chars with codepoints beyond 255) to
        stdout.
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
        row, col = divmod(char_code, 16)
        chars[row][col] = char
    if non_8bit and print_non_latin1:
        indices, *_ = zip(*non_8bit)
        max_indices_len = max(map(len, indices))
        print("The font face contains the following glyphs corresponding to "
              "code points beyond 0xff:")
        for index, char, name in non_8bit:
            print(f"{index:>{max_indices_len}} {char} {name}")

    fig, ax = plt.subplots(figsize=(8, 4))
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

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Display a font table.")
    parser.add_argument("path", nargs="?", help="Path to the font file.")
    parser.add_argument("--print-non-latin1", action="store_true",
                        help="Print non-latin1 chars to stdout.")
    args = parser.parse_args()

    draw_font_table(args.path, args.print_non_latin1)
