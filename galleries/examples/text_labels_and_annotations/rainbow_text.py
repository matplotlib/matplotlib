"""
============
Rainbow text
============

The example shows how to string together several text objects.

History
-------
On the matplotlib-users list back in February 2012, Gökhan Sever asked the
following question:

  | Is there a way in matplotlib to partially specify the color of a string?
  |
  | Example:
  |
  | plt.ylabel("Today is cloudy.")
  |
  | How can I show "today" as red, "is" as green and "cloudy." as blue?
  |
  | Thanks.

The solution below is modified from Paul Ivanov's original answer.
"""

import matplotlib.pyplot as plt


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs :
        All other keyword arguments are passed to plt.text() and plt.annotate(), so you
        can set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'horizontal':
        txt = ax.text(x, y, strings[0], color=colors[0], **kwargs)
        for s, c in zip(strings[1:], colors[1:]):
            txt = ax.annotate(' ' + s, xy=(1, 0), xycoords=txt,
                              va="bottom", color=c, **kwargs)

    elif orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')
        txt = ax.text(x, y, strings[0], color=colors[0], **kwargs)
        for s, c in zip(strings[1:], colors[1:]):
            txt = ax.annotate(' ' + s, xy=(0, 1), xycoords=txt,
                              va="bottom", color=c, **kwargs)


words = "all unicorns poop rainbows ! ! !".split()
colors = ['red', 'orange', 'gold', 'lawngreen', 'lightseagreen', 'royalblue',
          'blueviolet']
plt.figure(figsize=(8, 8))
rainbow_text(0.1, 0.05, words, colors, size=18)
rainbow_text(0.05, 0.1, words, colors, orientation='vertical', size=18)

plt.show()
