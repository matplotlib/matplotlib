"""
You can explicitly set which font family is picked up for a given font
style (e.g., 'serif', 'sans-serif', or 'monospace').

In the example below, we only allow one font family (Tahoma) for the
san-serif font style.  You the default family with the font.family rc
param, e.g.,::

  rcParams['font.family'] = 'sans-serif'

and for the font.family you set a list of font styles to try to find
in order::

  rcParams['font.sans-serif'] = ['Tahoma', 'Bitstream Vera Sans', 'Lucida Grande', 'Verdana']

"""

# -*- noplot -*-

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1,2,3], label='test')

ax.legend()
plt.show()

