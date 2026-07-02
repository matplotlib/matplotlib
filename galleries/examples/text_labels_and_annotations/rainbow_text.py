"""
==================================================
Concatenate text objects with different properties
==================================================

The example strings together several Text objects with different properties
(e.g., color or font), positioning each one after the other.  The first Text
is created directly using `~.Axes.text`; all subsequent ones are created with
`~.Axes.annotate`, which allows positioning the Text's lower left corner at the
lower right corner (``xy=(1, 0)``) of the previous one (``xycoords=text``).
"""

import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20

ax = plt.figure().add_subplot(xticks=[], yticks=[])

# The first word, created with text().
text = ax.text(.1, .5, "Matplotlib", color="red")
# Subsequent words, positioned with annotate(), relative to the preceding one.
text = ax.annotate(
    " says,", xycoords=text, xy=(1, 0), verticalalignment="bottom",
    color="gold", weight="bold")  # custom properties
text = ax.annotate(
    " hello", xycoords=text, xy=(1, 0), verticalalignment="bottom",
    color="green", style="italic")  # custom properties
text = ax.annotate(
    " world!", xycoords=text, xy=(1, 0), verticalalignment="bottom",
    color="blue", family="serif")  # custom properties

plt.show()

# %%
#
# Figure Text
# -----------
#
# Figure text can be concatenated in a similar manner by creating a `~.text.Annotation`
# object and adding it to the figure:


import matplotlib.text as mtext

fig, ax = plt.subplots(subplot_kw=dict(visible=False))

text = fig.text(.1, .5, "Matplotlib", color="red")

text = mtext.Annotation(" says,", xycoords=text, xy=(1, 0),
                        va="bottom", color="gold", weight="bold")
fig.add_artist(text)  # manually add artist to figure

text = mtext.Annotation(" hello", xycoords=text, xy=(1, 0),
                        va="bottom", color="green", style="italic")
fig.add_artist(text)

text = mtext.Annotation(" world!", xycoords=text, xy=(1, 0),
                        va="bottom", color="blue", family="serif")
fig.add_artist(text)

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.annotate`
#    - `matplotlib.text.Annotation`
#
# .. tags:: component: annotation, component: text, styling: color
