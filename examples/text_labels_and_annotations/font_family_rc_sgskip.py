"""
===========================
Configuring the font family
===========================

You can explicitly set which font family is picked up, either by specifying
family names of fonts installed on user's system, or generic-families
(e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or 'cursive'),
or a combination of both.
(see :doc:`font tutorial </tutorials/text/text_props>`)

In the example below, we are overriding the default sans-serif generic family
to include a specific (Tahoma) font. (Note that the best way to achieve this
would simply be to prepend 'Tahoma' in 'font.family')

The default family is set with the font.family rcparam,
e.g. ::

  rcParams['font.family'] = 'sans-serif'

and for the font.family you set a list of font styles to try to find
in order::

  rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                 'Lucida Grande', 'Verdana']
"""

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']

fig, ax = plt.subplots(figsize=(6, 1), facecolor="#eefade")
ax.text(0.01, 0.2, "Hello World! 01", size = 40)
ax.axis("off")
plt.show()


"""
And here a second example:
"""

plt.rcParams['font.family'] = 'monospace'
# plt.rcParams['font.monospace'] = ['Computer Modern Typewriter'] # this line gives an error currently, therefore the 
# question: How can one access the fonts from https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/mpl-data/matplotlibrc#L273 
# the correct way?
fig, ax = plt.subplots(figsize=(6, 1), facecolor="#eefade")
ax.text(0.01, 0.2, "Hello World! 01", size = 40)
ax.axis("off")
plt.show()