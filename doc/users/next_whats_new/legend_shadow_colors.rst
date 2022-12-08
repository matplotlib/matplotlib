Configurable legend shadows
---------------------------
The *shadow* parameter of legends now accepts dicts in addition to booleans.
Dictionaries can contain any keywords for `.patches.Patch`.
For example, this allows one to set the color and/or the transparency of a legend shadow:

.. code-block:: python

   ax.legend(loc='center left', shadow={'color': 'red', 'alpha': 0.5})

and to control the shadow location:

.. code-block:: python

   ax.legend(loc='center left', shadow={"ox":20, "oy":-20})

Configuration is currently not supported via :rc:`legend.shadow`.
