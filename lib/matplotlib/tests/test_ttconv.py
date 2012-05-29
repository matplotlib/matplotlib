import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

@image_comparison(baseline_images=["truetype-conversion"],
                  extensions=["pdf", "png"])
def test_truetype_conversion():
    matplotlib.rcParams['pdf.fonttype'] = 3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(0, 0, "ABCDE", name="mpltest", fontsize=80)
    ax.set_xticks([])
    ax.set_yticks([])
