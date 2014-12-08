#!/usr/bin/env python
"""
Same as fonts_demo using kwargs.  If you prefer a more pythonic, OO
style of coding, see examples/fonts_demo.py.

"""
from matplotlib.font_manager import FontProperties
from pylab import *

subplot(111, axisbg='w')
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}

# Show family options

families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']

t = text(-0.8, 0.9, 'family', size='large', **alignment)

yp = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

for k, family in enumerate(families):
    t = text(-0.8, yp[k], family, family=family, **alignment)

# Show style options

styles = ['normal', 'italic', 'oblique']

t = text(-0.4, 0.9, 'style', **alignment)

for k, style in enumerate(styles):
    t = text(-0.4, yp[k], style, family='sans-serif', style=style,
             **alignment)

# Show variant options

variants = ['normal', 'small-caps']

t = text(0.0, 0.9, 'variant', **alignment)

for k, variant in enumerate(variants):
    t = text(0.0, yp[k], variant, family='serif', variant=variant,
             **alignment)

# Show weight options

weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

t = text(0.4, 0.9, 'weight', **alignment)

for k, weight in enumerate(weights):
    t = text(0.4, yp[k], weight, weight=weight,
             **alignment)

# Show size options

sizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
         'x-large', 'xx-large']

t = text(0.8, 0.9, 'size', **alignment)

for k, size in enumerate(sizes):
    t = text(0.8, yp[k], size, size=size,
             **alignment)

x = -0.4
# Show bold italic
t = text(x, 0.1, 'bold italic', style='italic',
         weight='bold', size='x-small',
         **alignment)

t = text(x, 0.2, 'bold italic',
         style='italic', weight='bold', size='medium',
         **alignment)

t = text(x, 0.3, 'bold italic',
         style='italic', weight='bold', size='x-large',
         **alignment)

axis([-1, 1, 0, 1])

show()
