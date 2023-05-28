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
    if ax is None:
        ax = plt.gca()

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'horizontal':
        txt = ax.text(x, y, strings[0], color=colors[0], **kwargs)
        for s, c in zip(strings[1:], colors[1:]):
            t = ax.annotate(s, xy=(1, 0),
                            xycoords=txt, xytext=(.5, 0), textcoords="offset fontsize", va="bottom", color=c, **kwargs)
            txt = t

    elif orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')
        txt = ax.text(x, y, strings[0], color=colors[0], **kwargs)
        for s, c in zip(strings[1:], colors[1:]):
            t = ax.annotate(s, xy=(0, 1),
                            xycoords=txt, xytext=(0, .5), textcoords="offset fontsize", va="bottom", color=c, **kwargs)
            txt = t


words = "all unicorns poop rainbows ! ! !".split()
colors = ['red', 'orange', 'gold', 'lawngreen', 'lightseagreen', 'royalblue',
          'blueviolet']
plt.figure(figsize=(8, 8))
rainbow_text(0.1, 0.05, words, colors, size=18)
rainbow_text(0.05, 0.1, words, colors, orientation='vertical', size=18)

plt.show()
