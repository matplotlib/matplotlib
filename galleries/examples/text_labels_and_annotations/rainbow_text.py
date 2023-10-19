"""
====================================================
Concatenating text objects with different properties
====================================================

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
