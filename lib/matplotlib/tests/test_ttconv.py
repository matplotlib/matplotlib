from pathlib import Path

import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@image_comparison(["truetype-conversion.pdf"])
# mpltest.ttf does not have "l"/"p" glyphs so we get a warning when trying to
# get the font extents.
def test_truetype_conversion(recwarn):
    matplotlib.rcParams['pdf.fonttype'] = 3
    fig, ax = plt.subplots()
    ax.text(0, 0, "ABCDE",
            font=Path(__file__).with_name("mpltest.ttf"), fontsize=80)
    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(["ttconv_transforms"], extensions=["pdf"])
def test_ttconv_transforms():
    matplotlib.rcParams['pdf.fonttype'] = 3
    fig, ax = plt.subplots()
    kw = {'font': Path(__file__).with_name("FreeSerifSubset.ttf"), 'fontsize': 14}
    # characters where Free Serif uses various scales and linear transforms
    # e.g. the right paren is a reflected left paren
    ax.text(.1, .1, "parens, FAX, u with two diacritics: ()℻ǘ", **kw)
    ax.text(.1, .2, "double arrows LURD, single arrows LURD: ⇐⇑⇒⇓←↑→↓", **kw)
    ax.text(.1, .3, "corner arrows, wreath product: ↴↵≀", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
