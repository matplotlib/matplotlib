"""
===================================
Using a ttf font file in Matplotlib
===================================

Although it is usually not a good idea to explicitly point to a single ttf file
for a font instance, you can do so using the `font_manager.FontProperties`
*fname* argument.

Here, we use the Computer Modern roman font (``cmr10``) shipped with
Matplotlib.

For a more flexible solution, see :doc:`/gallery/api/font_family_rc_sgskip` and
:doc:`/gallery/text_labels_and_annotations/fonts_demo`.
"""

import os
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fpath = os.path.join(rcParams["datapath"], "fonts/ttf/cmr10.ttf")
prop = fm.FontProperties(fname=fpath)
fname = os.path.split(fpath)[1]
ax.set_title('This is a special font: {}'.format(fname), fontproperties=prop)
ax.set_xlabel('This is the default font')

plt.show()
