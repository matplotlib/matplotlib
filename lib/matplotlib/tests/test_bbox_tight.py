from matplotlib import rcParams, rcParamsDefault
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import numpy as np

@image_comparison(baseline_images=['bbox_inches_tight'], remove_text=True,
                  savefig_kwarg=dict(bbox_inches='tight'), tol=15)
def test_bbox_inches_tight():
    "Test that a figure saved using bbox_inches'tight' is clipped right"
    rcParams.update(rcParamsDefault)

    data = [[  66386,  174296,   75131,  577908,   32015],
            [  58230,  381139,   78045,   99308,  160454],
            [  89135,   80552,  152558,  497981,  603535],
            [  78415,   81858,  150656,  193263,   69638],
            [ 139361,  331509,  343164,  781380,   52269]]

    colLabels = rowLabels = [''] * 5

    rows = len(data)
    ind = np.arange(len(colLabels)) + 0.3  # the x locations for the groups
    cellText = []
    width = 0.4     # the width of the bars
    yoff = np.array([0.0] * len(colLabels))
    # the bottom values for stacked bar chart
    fig, ax = plt.subplots(1,1)
    for row in xrange(rows):
        plt.bar(ind, data[row], width, bottom=yoff)
        yoff = yoff + data[row]
        cellText.append([''])
    plt.xticks([])
    plt.legend([''] * 5, loc = (1.2, 0.2))
    # Add a table at the bottom of the axes
    cellText.reverse()
    the_table = plt.table(cellText=cellText,
                          rowLabels=rowLabels,
                          colLabels=colLabels, loc='bottom')

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
