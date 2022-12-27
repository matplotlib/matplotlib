r"""
===================================
Using a ttf font file in Matplotlib
===================================

Although it is usually not a good idea to explicitly point to a single ttf file
for a font instance, you can do so by passing a `pathlib.Path` instance as the
*font* parameter.  Note that passing paths as `str`\s is intentionally not
supported, but you can simply wrap `str`\s in `pathlib.Path`\s as needed.

Here, we use the Computer Modern roman font (``cmr10``) shipped with
Matplotlib.

For a more flexible solution, see
:doc:`/gallery/text_labels_and_annotations/font_family_rc` and
:doc:`/gallery/text_labels_and_annotations/fonts_demo`.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fpath = Path(mpl.get_data_path(), "fonts/ttf/cmr10.ttf")
ax.set_title(f'This is a special font: {fpath.name}', font=fpath)
ax.set_xlabel('This is the default font')

plt.show()


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.set_title`
