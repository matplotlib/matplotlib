``mathtext`` now supports ``\substack``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``\substack`` can be used to create multi-line subscripts or superscripts within an equation.

To use it to enclose the math in a substack command as shown:

.. code-block::

    r'$\sum_{\substack{1\leq i\leq 3\\ 1\leq j\leq 5}}$'

.. mathmpl::

    \sum_{\substack{1\leq i\leq 3\\ 1\leq j\leq 5}}


``mathtext`` now supports ``\middle`` delimiter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``\middle`` delimiter has been added, and can now be used with the
``\left`` and ``\right`` delimiters:

To use the middle command enclose it in between the ``\left`` and
``\right`` delimiter command as shown:

.. code-block::

    r'$\left( \frac{a}{b} \middle| q \right)$'

.. mathmpl::

    \left( \frac{a}{b} \middle| q \right)
