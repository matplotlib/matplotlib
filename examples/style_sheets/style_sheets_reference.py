"""
This example demonstrates different available style sheets on a common example.

The different plots are heavily similar to the other ones in the style sheet
gallery.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(ax, prng, nb_samples=200):
    """ Scatter plot.

    NB: `plt.scatter` doesn't use default colors.
    """
    x, y = prng.normal(size=(2, nb_samples))
    ax.plot(x, y, 'o')
    ax.set_xlabel('X-label')
    ax.set_ylabel('Y-label')
    return ax


def plot_colored_sinusoidal_lines(ax):
    """ Plot sinusoidal lines with colors from default color cycle.
    """
    L = 2*np.pi
    x = np.linspace(0, L)
    ncolors = len(plt.rcParams['axes.color_cycle'])
    shift = np.linspace(0, L, ncolors, endpoint=False)
    for s in shift:
        ax.plot(x, np.sin(x + s), '-')
    ax.margins(0)
    return ax


def plot_bar_graphs(ax, prng, min_value=5, max_value=25, nb_samples=5):
    """ Plot two bar graphs side by side, with letters as xticklabels.
    """
    x = np.arange(nb_samples)
    ya, yb = prng.randint(min_value, max_value, size=(2, nb_samples))
    width = 0.25
    ax.bar(x, ya, width)
    ax.bar(x + width, yb, width, color=plt.rcParams['axes.color_cycle'][2])
    ax.set_xticks(x + width)
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e'])
    return ax


def plot_colored_circles(ax, prng, nb_samples=15):
    """ Plot circle patches.

    NB: drawing a fixed amount of samples, rather than using the length of
    the color cycle, because different styles may have different numbers of
    colors.
    """
    list_of_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    max_idx = min(nb_samples, len(list_of_colors))
    for color in list_of_colors[0:max_idx]:
        ax.add_patch(plt.Circle(prng.normal(scale=3, size=2),
                     radius=1.0, color=color))
    # Force the limits to be the same accross the styles (because different
    # styles may have different numbers of available colors).
    ax.set_xlim([-5., 7.])
    ax.set_ylim([-8.5, 3.5])
    ax.set_aspect('equal', adjustable='box')  # to plot circles as circles
    return ax


def plot_image_and_patch(ax, prng, size=(20, 20)):
    """ Plot an image with random values and superimpose a circular patch.
    """
    values = prng.random_sample(size=size)
    ax.imshow(values, interpolation='none')
    c = plt.Circle((5, 5), radius=5, label='patch')
    ax.add_patch(c)


def plot_histograms(ax, prng, nb_samples=10000):
    """ Plot 4 histograms and a text annotation.
    """
    params = ((10, 10), (4, 12), (50, 12), (6, 55))
    for values in (prng.beta(a, b, size=nb_samples) for a, b in params):
        ax.hist(values, histtype="stepfilled", bins=30, alpha=0.8, normed=True)
    # Add a small annotation
    ax.annotate('Annotation', xy=(0.25, 4.25), xycoords='data',
                xytext=(0.4, 10), textcoords='data',
                bbox=dict(boxstyle="round", alpha=0.2),
                arrowprops=dict(
                          arrowstyle="->",
                          connectionstyle="angle,angleA=-95,angleB=35,rad=10"),
                )
    return ax


def plot_figure(style_label=None):
    """
    Setup and plot the demonstration figure with the style `style_label`.
    If `style_label`, fall back to the `default` style.
    """
    if style_label is None:
        style_label = 'default'

    # Use a dedicated RandomState instance to draw the same "random" values
    # across the different figures
    prng = np.random.RandomState(145236987)

    fig, axes = plt.subplots(ncols=3, nrows=2, num=style_label)
    fig.suptitle(style_label)

    axes_list = axes.ravel()  # for convenience
    plot_scatter(axes_list[0], prng)
    plot_image_and_patch(axes_list[1], prng)
    plot_bar_graphs(axes_list[2], prng)
    plot_colored_circles(axes_list[3], prng)
    plot_colored_sinusoidal_lines(axes_list[4])
    plot_histograms(axes_list[5], prng)

    return fig


if __name__ == "__main__":

    # Plot a demonstration figure for every available style sheet.
    for style_label in plt.style.available:
        with plt.style.context(style_label):
            fig = plot_figure(style_label=style_label)

    plt.show()
