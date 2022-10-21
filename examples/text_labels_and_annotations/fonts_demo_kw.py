"""
==============================
Fonts demo (keyword arguments)
==============================

Set font properties using keyword arguments.

See :doc:`fonts_demo` to achieve the same effect using setters.
"""

import matplotlib.pyplot as plt

fig = plt.figure()
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
yp = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# Show family options
fig.text(0.1, 0.9, 'family', size='large', **alignment)
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
for k, family in enumerate(families):
    fig.text(0.1, yp[k], family, family=family, **alignment)

# Show style options
fig.text(0.3, 0.9, 'style', **alignment)
styles = ['normal', 'italic', 'oblique']
for k, style in enumerate(styles):
    fig.text(0.3, yp[k], style, family='sans-serif', style=style, **alignment)

# Show variant options
fig.text(0.5, 0.9, 'variant', **alignment)
variants = ['normal', 'small-caps']
for k, variant in enumerate(variants):
    fig.text(0.5, yp[k], variant, family='serif', variant=variant, **alignment)

# Show weight options
fig.text(0.7, 0.9, 'weight', **alignment)
weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
for k, weight in enumerate(weights):
    fig.text(0.7, yp[k], weight, weight=weight, **alignment)

# Show size options
fig.text(0.9, 0.9, 'size', **alignment)
sizes = [
    'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
for k, size in enumerate(sizes):
    fig.text(0.9, yp[k], size, size=size, **alignment)

# Show bold italic
fig.text(0.3, 0.1, 'bold italic',
         style='italic', weight='bold', size='x-small', **alignment)
fig.text(0.3, 0.2, 'bold italic',
         style='italic', weight='bold', size='medium', **alignment)
fig.text(0.3, 0.3, 'bold italic',
         style='italic', weight='bold', size='x-large', **alignment)

plt.show()
