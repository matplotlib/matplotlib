#!/usr/bin/env python
from matplotlib.font_manager import set_default_font, get_default_font
from matplotlib.matlab import *

font = get_default_font()
font.set_family('serif')  # this operates on the default font!

otherFont = font.copy()   # this is an independent font
otherFont.set_family('monospace')
otherFont.set_weight('bold')
otherFont.set_size('larger')

plot(arange(20))
title('something')

set_default_font(otherFont)
for i in range(1,15,2):
    text(i, i, 'label %d'%i, color='g')     # uses otherFont
set_default_font(font)                     # restore default
xlabel('hi mom')                           # uses font
show()
