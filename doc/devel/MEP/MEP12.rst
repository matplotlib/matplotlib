=====================================
 MEP12: Improve Gallery and Examples
=====================================
.. contents::
   :local:


Status
======

**Progress**

Initial changes added in 1.3. Conversion of the gallery is on-going.
29 September 2015 - The last ``pylab_examples`` where `pylab` is imported has been converted over to use :mod:`matplotlib.pyplot` and `numpy`.

Branches and Pull requests
==========================

#1623, #1924, #2181

PR `#2474 <https://github.com/matplotlib/matplotlib/pull/2474>`_
demonstrates a single example being cleaned up and moved to the
appropriate section.

Abstract
========

Reorganizing the matplotlib plot gallery would greatly simplify
navigation of the gallery. In addition, examples should be cleaned-up
and simplified for clarity.


Detailed description
====================

The matplotlib gallery was recently set up to split examples up into
sections.  As discussed in that PR [1]_, the current example sections
(``api``, ``pylab_examples``) aren't terribly useful to users: New
sections in the gallery would help users find relevant examples.

These sections would also guide a cleanup of the examples: Initially,
all the current examples would remain and be listed under their
current directories.  Over time, these examples could be cleaned up
and moved into one of the new sections.

This process allows users to easily identify examples that need to be
cleaned up; i.e. anything in the ``api`` and ``pylab_examples``
directories.


Implementation
==============

1. Create new gallery sections. [Done]
2. Clean up examples and move them to the new gallery sections (over the course
   of many PRs and with the help of many users/developers). [In progress]

Gallery sections
----------------

The naming of sections is critical and will guide the clean-up
effort. The current sections are:

* Lines, bars, and markers (more-or-less 1D data)
* Shapes and collections
* Statistical plots
* Images, contours, and fields
* Pie and polar charts: Round things
* Color
* Text, labels, and annotations
* Ticks and spines
* Subplots, axes, and figures
* Specialty plots (e.g., sankey, radar, tornado)
* Showcase (plots with tweaks to make them publication-quality)
* separate sections for toolboxes (already exists: 'mplot3d',
  'axes_grid', 'units', 'widgets')

These names are certainly up for debate. As these sections grow, we
should reevaluate them and split them up as necessary.


Clean up guidelines
-------------------

The current examples in the ``api`` and ``pylab_examples`` sections of
the gallery would remain in those directories until they are cleaned
up. After clean-up, they would be moved to one of the new gallery
sections described above. "Clean-up" should involve:

* `sphinx-gallery docstrings <https://sphinx-gallery.readthedocs.io/en/latest/>`_:
  a title and a description of the example formatted as follows, at the top of
  the example::

    """
    ===============================
    Colormaps alter your perception
    ===============================

    Here I plot the function

    .. math:: f(x, y) = \sin(x) + \cos(y)

    with different colormaps. Look at how colormaps alter your perception!
    """


* PEP8_ clean-ups (running `flake8
  <https://pypi.org/project/flake8>`_, or a similar checker, is
  highly recommended)
* Commented-out code should be removed.
* Replace uses of `pylab` interface with `.pyplot` (+ `numpy`,
  etc.). See `c25ef1e
  <https://github.com/tonysyu/matplotlib/commit/c25ef1e02b3a0ecb279492409dac0de9b3d2c0e2>`_
* Remove shebang line, e.g.:

      #!/usr/bin/env python

* Use consistent imports. In particular:

      import numpy as np

      import matplotlib.pyplot as plt

  Avoid importing specific functions from these modules (e.g. ``from
  numpy import sin``)

* Each example should focus on a specific feature (excluding
  ``showcase`` examples, which will show more "polished"
  plots). Tweaking unrelated to that feature should be removed. See
  `f7b2217
  <https://github.com/tonysyu/matplotlib/commit/f7b2217a1f92343e8aca0684d19c9915cc2e8674>`_,
  `e57b5fc
  <https://github.com/tonysyu/matplotlib/commit/e57b5fc31fbad83ed9c43c77ef15368efdcb9ec1>`_,
  and `1458aa8
  <https://github.com/tonysyu/matplotlib/commit/1458aa87c5eae9dd99e141956a6adf7a0f3c6707>`_

Use of `pylab` should be demonstrated/discussed on a dedicated help
page instead of the gallery examples.

**Note:** When moving an existing example, you should search for
references to that example.  For example, the API documentation for
:file:`axes.py` and :file:`pyplot.py` may use these examples to generate
plots. Use your favorite search tool (e.g., grep, ack, `grin
<https://pypi.org/project/grin>`_, `pss
<https://pypi.org/project/pss>`_) to search the matplotlib
package. See `2dc9a46
<https://github.com/tonysyu/matplotlib/commit/2dc9a4651e5e566afc0866c603aa8d06aaf32b71>`_
and `aa6b410
<https://github.com/tonysyu/matplotlib/commit/aa6b410f9fa085ccf5f4f962a6f26af5beeae7af>`_


Additional suggestions
~~~~~~~~~~~~~~~~~~~~~~

* Provide links (both ways) between examples and API docs for the
  methods/objects used. (issue `#2222
  <https://github.com/matplotlib/matplotlib/issues/2222>`_)
* Use ``plt.subplots`` (note trailing "s") in preference over
  ``plt.subplot``.
* Rename the example to clarify it's purpose. For example, the most
  basic demo of ``imshow`` might be ``imshow_demo.py``, and one
  demonstrating different interpolation settings would be
  ``imshow_demo_interpolation.py`` (*not* ``imshow_demo2.py``).
* Split up examples that try to do too much. See `5099675
  <https://github.com/tonysyu/matplotlib/commit/509967518ce5ce5ba31edf12486ffaa344e748f2>`_
  and `fc2ab07
  <https://github.com/tonysyu/matplotlib/commit/fc2ab07cc586abba4c024d8c0d841c4357a3936f>`_
* Delete examples that don't show anything new.
* Some examples exercise esoteric features for unit testing. These
  tweaks should be moved out of the gallery to an example in the
  ``unit`` directory located in the root directory of the package.
* Add plot titles to clarify intent of the example. See `bd2b13c
  <https://github.com/tonysyu/matplotlib/commit/bd2b13c54bf4aa2058781b9a805d68f2feab5ba5>`_


Backward compatibility
======================

The website for each Matplotlib version is readily accessible, so
users who want to refer to old examples can still do so.


Alternatives
============

Tags
----

Tagging examples will also help users search the example gallery. Although tags
would be a big win for users with specific goals, the plot gallery will remain
the entry point to these examples, and sections could really help users
navigate the gallery. Thus, tags are complementary to this reorganization.


.. _PEP8: https://www.python.org/dev/peps/pep-0008/

.. [1] https://github.com/matplotlib/matplotlib/pull/714
