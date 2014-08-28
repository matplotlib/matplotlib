# -*- noplot -*-
"""
Although it is usually not a good idea to explicitly point to a single
ttf file for a font instance, you can do so using the
font_manager.FontProperties fname argument (for a more flexible
solution, see the font_fmaily_rc.py and fonts_demo.py examples).
"""
import sys
import os
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])

if sys.platform == 'win32':
    fpath = 'C:\\Windows\\Fonts\\Tahoma.ttf'
elif sys.platform.startswith('linux'):
    basedir = '/usr/share/fonts/truetype'
    fonts = ['freefont/FreeSansBoldOblique.ttf',
             'ttf-liberation/LiberationSans-BoldItalic.ttf',
             'msttcorefonts/Comic_Sans_MS.ttf']
    for fpath in fonts:
        if os.path.exists(os.path.join(basedir, fpath)):
            break
else:
    fpath = '/Library/Fonts/Tahoma.ttf'

if os.path.exists(fpath):
    prop = fm.FontProperties(fname=fpath)
    fname = os.path.split(fpath)[1]
    ax.set_title('this is a special font: %s' % fname, fontproperties=prop)
else:
    ax.set_title('Demo fails--cannot find a demo font')
ax.set_xlabel('This is the default font')

plt.show()
