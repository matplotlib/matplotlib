import numpy as np

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@image_comparison(baseline_images=['color_cycle_basic'], remove_text=True)
def test_colorcycle_basic():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle(['r', 'g', 'y'])
    xs = np.arange(10)
    ys = 0.25 * xs + 2
    # Should be red
    ax.plot(xs, ys)
    # Should be green
    ys = 0.45 * xs + 3
    ax.plot(xs, ys)
    # Should be yellow
    ys = 0.65 * xs + 4
    ax.plot(xs, ys)
    # Should be red
    ys = 0.85 * xs + 5
    ax.plot(xs, ys)


@image_comparison(baseline_images=['linestyle_cycle_basic'], remove_text=True)
def test_linestylecycle_basic():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_linestyle_cycle(['-', '--', ':'])
    xs = np.arange(10)
    # Should be solid
    ys = 0.25 * xs + 2
    ax.plot(xs, ys)
    # Should be dashed
    ys = 0.45 * xs + 3
    ax.plot(xs, ys)
    # Should be dotted
    ys = 0.65 * xs + 4
    ax.plot(xs, ys)
    # Should be solid
    ys = 0.85 * xs + 5
    ax.plot(xs, ys)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
