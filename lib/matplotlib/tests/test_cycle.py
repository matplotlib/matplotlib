import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@image_comparison(baseline_images=['set_cycles'])
def test_set_cycles():
    x = np.linspace(0, 2 * np.pi)
    offsets = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    yy = np.transpose([np.sin(x + phi) for phi in offsets])
    fig, ax1 = plt.subplots(nrows=1)
    ax1.set_color_cycle(['c', 'm', 'y', 'k'])
    ax1.set_cycle('linestyle', ['--', '-.', ':'])
    ax1.set_cycle('linewidth', [3, 1])
    ax1.set_cycle('marker', ['>', '<', 'o'])
    ax1.set_cycle('markersize', [5, 10])
    ax1.set_cycle('markerfacecolor', ['black'])
    ax1.set_cycle('markeredgecolor', ['blue'])
    ax1.set_cycle('markeredgewidth', [1])
    ax1.set_cycle('antialiased', [True, False])
    ax1.set_cycle('dash_joinstyle',  ['miter', 'round', 'bevel'])
    ax1.set_cycle('solid_joinstyle', ['miter', 'round', 'bevel'])
    ax1.set_cycle('dash_capstyle', ['butt', 'round', 'projecting'])
    ax1.set_cycle('solid_capstyle', ['butt', 'round', 'projecting'])
    ax1.plot(yy)


@image_comparison(baseline_images=['clear_one_cycle'])
def test_clear_one_cycle():
    x = np.linspace(0, 2 * np.pi)
    offsets = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    yy = np.transpose([np.sin(x + phi) for phi in offsets])
    fig, ax1 = plt.subplots(nrows=1)
    ax1.set_color_cycle(['c', 'm', 'y', 'k'])
    ax1.set_cycle('linestyle', ['--', '-.', ':'])
    ax1.set_cycle('linewidth', [3, 1])
    ax1.clear_cycle('linestyle')
    ax1.plot(yy)


@image_comparison(baseline_images=['clear_all_cycle'])
def test_clear_all_cycle():
    x = np.linspace(0, 2 * np.pi)
    offsets = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    yy = np.transpose([np.sin(x + phi) for phi in offsets])
    fig, ax1 = plt.subplots(nrows=1)
    ax1.set_color_cycle(['c', 'm', 'y', 'k'])
    ax1.set_cycle('linestyle', ['--', '-.', ':'])
    ax1.set_cycle('linewidth', [3, 1])
    ax1.clear_all_cycle()
    ax1.plot(yy)
