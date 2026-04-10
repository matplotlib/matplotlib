``axes.prop_cycle`` rcParam security improvements
-------------------------------------------------

The ``axes.prop_cycle`` rcParam is now parsed in a safer and more restricted
manner. Only literals, ``cycler()`` and ``concat()`` calls, the operators
``+`` and ``*``, and slicing are allowed. All previously valid cycler strings
documented at https://matplotlib.org/cycler/ are still supported, for example:

.. code-block:: none

   axes.prop_cycle : cycler('color', ['r', 'g', 'b']) + cycler('linewidth', [1, 2, 3])
   axes.prop_cycle : 2 * cycler('color', 'rgb')
   axes.prop_cycle : concat(cycler('color', 'rgb'), cycler('color', 'cmk'))
   axes.prop_cycle : cycler('color', 'rgbcmk')[:3]
