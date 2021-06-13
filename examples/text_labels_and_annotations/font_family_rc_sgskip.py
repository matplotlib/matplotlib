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

fig, ax = plt.subplots()
ax.plot([1, 2, 3], label='test')

ax.legend()
plt.show()
