#!/usr/bin/env python

from matplotlib.font_manager import fontManager, get_default_font, set_default_font
from matplotlib.matlab import *

subplot(111, axisbg='w')

font0 = get_default_font()

###  Show family options

#family = ['serif', 'sans-serif', 'cursive', 'serif', 'monospace']
family = ['serif', 'sans-serif', 'monospace']

font1 = font0.copy()
font1.set_size('large')

t = text(-0.8, 0.9, 'family', fontproperties=font1,
         horizontalalignment='center', verticalalignment='center')

yp = [0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5]

for k in range(3):
    font = font0.copy()
    font.set_family(family[k])
    t = text(-0.8, yp[k], family[k], fontproperties=font,
             horizontalalignment='center', verticalalignment='center')

###  Show style options

style  = ['normal', 'italic', 'oblique']

t = text(-0.4, 0.9, 'style', fontproperties=font1,
         horizontalalignment='center', verticalalignment='center')

for k in range(2):
    font = font0.copy()
    font.set_family('sans-serif')
    font.set_style(style[k])
    t = text(-0.4, yp[k], style[k], fontproperties=font,
             horizontalalignment='center', verticalalignment='center')

###  Show variant options

variant= ['normal', 'small-caps']

t = text(0.0, 0.9, 'variant', fontproperties=font1,
         horizontalalignment='center', verticalalignment='center')

for k in range(1):
    font = font0.copy()
    font.set_family('serif')
    font.set_variant(variant[k])
    t = text( 0.0, yp[k], variant[k], fontproperties=font,
             horizontalalignment='center', verticalalignment='center')

###  Show weight options

weight = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

t = text( 0.4, 0.9, 'weight', fontproperties=font1,
         horizontalalignment='center', verticalalignment='center')

for k in range(7):
    font = font0.copy()
    font.set_weight(weight[k])
    t = text( 0.4, yp[k], weight[k], fontproperties=font,
             horizontalalignment='center', verticalalignment='center')

###  Show size options

size  = ['xx-small', 'x-small', 'small', 'medium', 'large',
         'x-large', 'xx-large']

t = text( 0.8, 0.9, 'size', fontproperties=font1,
         horizontalalignment='center', verticalalignment='center')

for k in range(7):
    font = font0.copy()
    font.set_size(size[k])
    t = text( 0.8, yp[k], size[k], fontproperties=font,
             horizontalalignment='center', verticalalignment='center')

###  Show bold italic

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('x-small')
t = text(-0.2, 0.1, 'bold italic', fontproperties=font,
         horizontalalignment='center', verticalalignment='center')

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('medium')
t = text(-0.2, -0.1, 'bold italic', fontproperties=font,
         horizontalalignment='center', verticalalignment='center')

font = font0.copy()
font.set_style('italic')
font.set_weight('bold')
font.set_size('x-large')
t = text(-0.2, -0.3, 'bold italic', fontproperties=font,
         horizontalalignment='center', verticalalignment='center')

show()
