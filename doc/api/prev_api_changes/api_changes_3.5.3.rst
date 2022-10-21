API Changes for 3.5.3
=====================

.. contents::
   :local:
   :depth: 1

Passing *linefmt* positionally is undeprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Positional use of all formatting parameters in `~.Axes.stem` has been
deprecated since Matplotlib 3.5. This deprecation is relaxed so that one can
still pass *linefmt* positionally, i.e. ``stem(x, y, 'r')``.
