"""
===================
Fonts demo (kwargs)
===================

Set font properties using kwargs.

See :doc:`fonts_demo` to achieve the same effect using setters.
"""

import matplotlib.pyplot as plt

alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}

# Show family options

families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']

t = plt.figtext(0.1, 0.9, 'family', size='large', **alignment)

yp = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

for k, family in enumerate(families):
    t = plt.figtext(0.1, yp[k], family, family=family, **alignment)

# Show style options

styles = ['normal', 'italic', 'oblique']

t = plt.figtext(0.3, 0.9, 'style', **alignment)

for k, style in enumerate(styles):
    t = plt.figtext(0.3, yp[k], style, family='sans-serif', style=style,
                    **alignment)

# Show variant options

variants = ['normal', 'small-caps']

t = plt.figtext(0.5, 0.9, 'variant', **alignment)

for k, variant in enumerate(variants):
    t = plt.figtext(0.5, yp[k], variant, family='serif', variant=variant,
                    **alignment)

# Show weight options

weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']

t = plt.figtext(0.7, 0.9, 'weight', **alignment)

for k, weight in enumerate(weights):
    t = plt.figtext(0.7, yp[k], weight, weight=weight, **alignment)

# Show size options

sizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
         'x-large', 'xx-large']

t = plt.figtext(0.9, 0.9, 'size', **alignment)

for k, size in enumerate(sizes):
    t = plt.figtext(0.9, yp[k], size, size=size, **alignment)

# Show bold italic
t = plt.figtext(0.3, 0.1, 'bold italic', style='italic',
                weight='bold', size='x-small',
                **alignment)
t = plt.figtext(0.3, 0.2, 'bold italic',
                style='italic', weight='bold', size='medium',
                **alignment)
t = plt.figtext(0.3, 0.3, 'bold italic',
                style='italic', weight='bold', size='x-large',
                **alignment)

plt.show()
