

import matplotlib.pyplot as plt


def test_check_figures_equal():

    res1 = plt.triplot([0, 2, 1], [0, 0, 1], [[0, 1, 2]], ls='--')
    res2 = plt.triplot([0, 2, 1], [0, 0, 1], [[0, 1, 2]], linestyle='--')
