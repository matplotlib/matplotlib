import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison


@image_comparison(baseline_images=['table_zorder'],
                  extensions=['png'],
                  remove_text=True)
def test_zorder():
    data = [[66386, 174296],
            [58230, 381139]]

    colLabels = ('Freeze', 'Wind')
    rowLabels = ['%d year' % x for x in (100, 50)]

    cellText = []
    yoff = np.array([0.0] * len(colLabels))
    for row in reversed(data):
        yoff += row
        cellText.append(['%1.1f' % (x/1000.0) for x in yoff])

    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(t, np.cos(t), lw=4, zorder=2)

    plt.table(cellText=cellText,
              rowLabels=rowLabels,
              colLabels=colLabels,
              loc='center',
              zorder=-2,
              )

    plt.table(cellText=cellText,
              rowLabels=rowLabels,
              colLabels=colLabels,
              loc='upper center',
              zorder=4,
              )
    plt.yticks([])
