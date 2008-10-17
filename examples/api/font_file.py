# -*- noplot -*-
"""
Although it is usually not a good idea to explicitly point to a single
ttf file for a font instance, you can do so using the
font_manager.FontProperties fname argument (for a more flexible
solution, see the font_fmaily_rc.py and fonts_demo.py examples).
"""
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3])

prop = fm.FontProperties(fname='/Library/Fonts/Tahoma.ttf')
ax.set_title('this is a special font', fontproperties=prop)
plt.show()

