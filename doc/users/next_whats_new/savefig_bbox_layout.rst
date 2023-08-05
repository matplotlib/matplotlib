pad_inches="layout" for savefig
-------------------------------

When using constrained or compressed layout,

.. code-block:: python

    savefig(filename, bbox_inches="tight", pad_inches="layout")

will now use the padding sizes defined on the layout engine.
