``mathtext`` now supports units for the bar thickness ``\genfrac`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This follows the standard LaTeX version where units are required.

.. code-block:: python

import matplotlib.pyplot as plt

plt.text(0.5, 0.5, r'$\genfrac{(}{)}{0.5cm}{0}{foo}{bar}$')
plt.draw()
