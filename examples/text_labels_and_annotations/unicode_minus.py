"""
=============
Unicode minus
=============

By default, tick labels at negative values are rendered using a `Unicode
minus`__ (U+2212) rather than an ASCII hyphen (U+002D).  This can be controlled
by setting :rc:`axes.unicode_minus` (which defaults to True).

__ https://en.wikipedia.org/wiki/Plus_and_minus_signs#Character_codes

This example showcases the difference between the two glyphs.
"""

import matplotlib.pyplot as plt

fig = plt.figure()
fig.text(.5, .5, "Unicode minus: \N{MINUS SIGN}1", horizontalalignment="right")
fig.text(.5, .4, "ASCII hyphen: -1", horizontalalignment="right")
plt.show()
