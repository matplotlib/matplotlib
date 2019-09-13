from pathlib import Path

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@image_comparison(["truetype-conversion.pdf"])
# mpltest.ttf does not have "l"/"p" glyphs so we get a warning when trying to
# get the font extents.
def test_truetype_conversion(recwarn):
    fontprop = FontProperties(
        fname=str(Path(__file__).with_name('mpltest.ttf').resolve()), size=80)
    matplotlib.rcParams['pdf.fonttype'] = 3
    fig, ax = plt.subplots()
    ax.text(0, 0, "ABCDE", fontproperties=fontprop)
    ax.set_xticks([])
    ax.set_yticks([])
