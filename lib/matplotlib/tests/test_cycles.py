from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import numpy as np

from cycler import cycler

@image_comparison(baseline_images=['color_cycle_basic'], remove_text=True,
                  extensions=['png'])
def test_colorcycle_basic():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color', ['r', 'g', 'y']))
    xs = np.arange(10)
    ys = 0.25 * xs + 2
    ax.plot(xs, ys, label='red', lw=4)
    ys = 0.45 * xs + 3
    ax.plot(xs, ys, label='green', lw=4)
    ys = 0.65 * xs + 4
    ax.plot(xs, ys, label='yellow', lw=4)
    ys = 0.85 * xs + 5
    ax.plot(xs, ys, label='red2', lw=4)
    ax.legend(loc='upper left')


@image_comparison(baseline_images=['lineprop_cycle_basic'], remove_text=True,
                  extensions=['png'])
def test_linestylecycle_basic():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('linestyle', ['-', '--', ':']))
    xs = np.arange(10)
    ys = 0.25 * xs + 2
    ax.plot(xs, ys, label='solid', lw=4)
    ys = 0.45 * xs + 3
    ax.plot(xs, ys, label='dashed', lw=4)
    ys = 0.65 * xs + 4
    ax.plot(xs, ys, label='dotted', lw=4)
    ys = 0.85 * xs + 5
    ax.plot(xs, ys, label='solid2', lw=4)
    ax.legend(loc='upper left')

@image_comparison(baseline_images=['fill_cycle_basic'], remove_text=True,
                  extensions=['png'])
def test_fillcycle_basic():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color',  ['red', 'green', 'yellow']))# +
                      #cycler('hatch', ['xx', 'O', '|-']))
    xs = np.arange(10)
    ys = 0.25 * xs**.5 + 2
    ax.fill(xs, ys, label='red, x')
    ys = 0.45 * xs**.5 + 3
    ax.fill(xs, ys, label='green, circle')
    ys = 0.65 * xs**.5 + 4
    ax.fill(xs, ys, label='yellow, cross')
    ys = 0.85 * xs**.5 + 5
    ax.fill(xs, ys, label='red2, x2')
    ax.legend(loc='upper left')


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)

