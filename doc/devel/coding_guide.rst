.. _coding_guidelines:

*****************
Coding guidelines
*****************

We appreciate these guidelines being followed because it improves the readability,
consistency, and maintainability of the code base.

.. admonition:: API guidelines
    :class: seealso

    If adding new features, changing behavior or function signatures, or removing
    public interfaces, please consult the :ref:`api_changes`.

.. _code-style:

PEP8, as enforced by flake8
===========================

Formatting should follow the recommendations of PEP8_, as enforced by flake8_.
Matplotlib modifies PEP8 to extend the maximum line length to 88
characters. You can check flake8 compliance from the command line with ::

    python -m pip install flake8
    flake8 /path/to/module.py

or your editor may provide integration with it.  Note that Matplotlib intentionally
does not use the black_ auto-formatter (1__), in particular due to its inability
to understand the semantics of mathematical expressions (2__, 3__).

.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _flake8: https://flake8.pycqa.org/
.. _black: https://black.readthedocs.io/
.. __: https://github.com/matplotlib/matplotlib/issues/18796
.. __: https://github.com/psf/black/issues/148
.. __: https://github.com/psf/black/issues/1984


Package imports
===============

Import the following modules using the standard scipy conventions::

  import numpy as np
  import numpy.ma as ma
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import matplotlib.cbook as cbook
  import matplotlib.patches as mpatches

In general, Matplotlib modules should **not** import `.rcParams` using ``from
matplotlib import rcParams``, but rather access it as ``mpl.rcParams``.  This
is because some modules are imported very early, before the `.rcParams`
singleton is constructed.

Variable names
==============

When feasible, please use our internal variable naming convention for objects
of a given class and objects of any child class:

+------------------------------------+---------------+------------------------------------------+
|             base class             | variable      |                multiples                 |
+====================================+===============+==========================================+
| `~matplotlib.figure.FigureBase`    | ``fig``       |                                          |
+------------------------------------+---------------+------------------------------------------+
| `~matplotlib.axes.Axes`            | ``ax``        |                                          |
+------------------------------------+---------------+------------------------------------------+
| `~matplotlib.transforms.Transform` | ``trans``     | ``trans_<source>_<target>``              |
+                                    +               +                                          +
|                                    |               | ``trans_<source>`` when target is screen |
+------------------------------------+---------------+------------------------------------------+

Generally, denote more than one instance of the same class by adding suffixes to
the variable names. If a format isn't specified in the table, use numbers or
letters as appropriate.

.. _type-hints:

Type hints
==========

If you add new public API or change public API, update or add the
corresponding `mypy <https://mypy.readthedocs.io/en/latest/>`_ type hints.
We generally use `stub files
<https://typing.readthedocs.io/en/latest/source/stubs.html#type-stubs>`_
(``*.pyi``) to store the type information; for example ``colors.pyi`` contains
the type information for ``colors.py``. A notable exception is ``pyplot.py``,
which is type hinted inline.

Type hints are checked by the mypy :ref:`pre-commit hook <pre-commit-hooks>`,
can often be verified by running ``tox -e stubtest``.

New modules and files: installation
===================================

* If you have added new files or directories, or reorganized existing ones, make sure the
  new files are included in the :file:`meson.build` in the corresponding directories.
* New modules *may* be typed inline or using parallel stub file like existing modules.

C/C++ extensions
================

* Extensions may be written in C or C++.

* Code style should conform to PEP7 (understanding that PEP7 doesn't
  address C++, but most of its admonitions still apply).

* Python/C interface code should be kept separate from the core C/C++
  code.  The interface code should be named :file:`FOO_wrap.cpp` or
  :file:`FOO_wrapper.cpp`.

* Header file documentation (aka docstrings) should be in Numpydoc
  format.  We don't plan on using automated tools for these
  docstrings, and the Numpydoc format is well understood in the
  scientific Python community.

* C/C++ code in the :file:`extern/` directory is vendored, and should be kept
  close to upstream whenever possible.  It can be modified to fix bugs or
  implement new features only if the required changes cannot be made elsewhere
  in the codebase.  In particular, avoid making style fixes to it.

.. _keyword-argument-processing:

Keyword argument processing
===========================

Matplotlib makes extensive use of ``**kwargs`` for pass-through customizations
from one function to another.  A typical example is
`~matplotlib.axes.Axes.text`.  The definition of `matplotlib.pyplot.text` is a
simple pass-through to `matplotlib.axes.Axes.text`::

  # in pyplot.py
  def text(x, y, s, fontdict=None, **kwargs):
      return gca().text(x, y, s, fontdict=fontdict, **kwargs)

`matplotlib.axes.Axes.text` (simplified for illustration) just
passes all ``args`` and ``kwargs`` on to ``matplotlib.text.Text.__init__``::

  # in axes/_axes.py
  def text(self, x, y, s, fontdict=None, **kwargs):
      t = Text(x=x, y=y, text=s, **kwargs)

and ``matplotlib.text.Text.__init__`` (again, simplified)
just passes them on to the `matplotlib.artist.Artist.update` method::

  # in text.py
  def __init__(self, x=0, y=0, text='', **kwargs):
      super().__init__()
      self.update(kwargs)

``update`` does the work looking for methods named like
``set_property`` if ``property`` is a keyword argument.  i.e., no one
looks at the keywords, they just get passed through the API to the
artist constructor which looks for suitably named methods and calls
them with the value.

As a general rule, the use of ``**kwargs`` should be reserved for
pass-through keyword arguments, as in the example above.  If all the
keyword args are to be used in the function, and not passed
on, use the key/value keyword args in the function definition rather
than the ``**kwargs`` idiom.

In some cases, you may want to consume some keys in the local
function, and let others pass through.  Instead of popping arguments to
use off ``**kwargs``, specify them as keyword-only arguments to the local
function.  This makes it obvious at a glance which arguments will be
consumed in the function.  For example, in
:meth:`~matplotlib.axes.Axes.plot`, ``scalex`` and ``scaley`` are
local arguments and the rest are passed on as
:meth:`~matplotlib.lines.Line2D` keyword arguments::

  # in axes/_axes.py
  def plot(self, *args, scalex=True, scaley=True, **kwargs):
      lines = []
      for line in self._get_lines(*args, **kwargs):
          self.add_line(line)
          lines.append(line)

.. _using_logging:

Using logging for debug messages
================================

Matplotlib uses the standard Python `logging` library to write verbose
warnings, information, and debug messages. Please use it! In all those places
you write `print` calls to do your debugging, try using `logging.debug`
instead!


To include `logging` in your module, at the top of the module, you need to
``import logging``.  Then calls in your code like::

  _log = logging.getLogger(__name__)  # right after the imports

  # code
  # more code
  _log.info('Here is some information')
  _log.debug('Here is some more detailed information')

will log to a logger named ``matplotlib.yourmodulename``.

If an end-user of Matplotlib sets up `logging` to display at levels more
verbose than ``logging.WARNING`` in their code with the Matplotlib-provided
helper::

  plt.set_loglevel("debug")

or manually with ::

  import logging
  logging.basicConfig(level=logging.DEBUG)
  import matplotlib.pyplot as plt

Then they will receive messages like

.. code-block:: none

   DEBUG:matplotlib.backends:backend MacOSX version unknown
   DEBUG:matplotlib.yourmodulename:Here is some information
   DEBUG:matplotlib.yourmodulename:Here is some more detailed information

Avoid using pre-computed strings (``f-strings``, ``str.format``,etc.) for logging because
of security and performance issues, and because they interfere with style handlers. For
example, use ``_log.error('hello %s', 'world')``  rather than ``_log.error('hello
{}'.format('world'))`` or ``_log.error(f'hello {s}')``.

Which logging level to use?
---------------------------

There are five levels at which you can emit messages.

- `logging.critical` and `logging.error` are really only there for errors that
  will end the use of the library but not kill the interpreter.
- `logging.warning` and `._api.warn_external` are used to warn the user,
  see below.
- `logging.info` is for information that the user may want to know if the
  program behaves oddly. They are not displayed by default. For instance, if
  an object isn't drawn because its position is ``NaN``, that can usually
  be ignored, but a mystified user could call
  ``logging.basicConfig(level=logging.INFO)`` and get an error message that
  says why.
- `logging.debug` is the least likely to be displayed, and hence can be the
  most verbose.  "Expected" code paths (e.g., reporting normal intermediate
  steps of layouting or rendering) should only log at this level.

By default, `logging` displays all log messages at levels higher than
``logging.WARNING`` to `sys.stderr`.

The `logging tutorial`_ suggests that the difference between `logging.warning`
and `._api.warn_external` (which uses `warnings.warn`) is that
`._api.warn_external` should be used for things the user must change to stop
the warning (typically in the source), whereas `logging.warning` can be more
persistent. Moreover, note that `._api.warn_external` will by default only
emit a given warning *once* for each line of user code, whereas
`logging.warning` will display the message every time it is called.

By default, `warnings.warn` displays the line of code that has the ``warn``
call. This usually isn't more informative than the warning message itself.
Therefore, Matplotlib uses `._api.warn_external` which uses `warnings.warn`,
but goes up the stack and displays the first line of code outside of
Matplotlib. For example, for the module::

    # in my_matplotlib_module.py
    import warnings

    def set_range(bottom, top):
        if bottom == top:
            warnings.warn('Attempting to set identical bottom==top')

running the script::

    from matplotlib import my_matplotlib_module
    my_matplotlib_module.set_range(0, 0)  # set range

will display

.. code-block:: none

    UserWarning: Attempting to set identical bottom==top
    warnings.warn('Attempting to set identical bottom==top')

Modifying the module to use `._api.warn_external`::

    from matplotlib import _api

    def set_range(bottom, top):
        if bottom == top:
            _api.warn_external('Attempting to set identical bottom==top')

and running the same script will display

.. code-block:: none

   UserWarning: Attempting to set identical bottom==top
   my_matplotlib_module.set_range(0, 0)  # set range

.. _logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial


.. _licence-coding-guide:

.. include:: license.rst
  :start-line: 2

.. toctree::
   :hidden:

   license.rst
