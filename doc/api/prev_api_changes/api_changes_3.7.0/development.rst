Development changes
-------------------


Windows wheel runtime bundling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wheels built for Windows now bundle the MSVC runtime DLL ``msvcp140.dll``. This
enables importing Matplotlib on systems that do not have the runtime installed.


Increase to minimum supported versions of dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


For Matplotlib 3.7, the :ref:`minimum supported versions <dependencies>` are
being bumped:

+------------+-----------------+---------------+
| Dependency |  min in mpl3.6  | min in mpl3.7 |
+============+=================+===============+
|   NumPy    |       1.19      |      1.20     |
+------------+-----------------+---------------+
| pyparsing  |      2.2.1      |    2.3.1      |
+------------+-----------------+---------------+
|     Qt     |                 |     5.10      |
+------------+-----------------+---------------+

- There are no wheels or conda packages that support both Qt 5.9 (or older) and
  Python 3.8 (or newer).

This is consistent with our :ref:`min_deps_policy` and `NEP29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`__


New dependencies
~~~~~~~~~~~~~~~~

* `importlib-resources <https://pypi.org/project/importlib-resources/>`_
  (>= 3.2.0; only required on Python < 3.10)

Maximum line length increased to 88 characters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The maximum line length for new contributions has been extended from 79 characters to
88 characters.
This change provides an extra 9 characters to allow code which is a single idea to fit
on fewer lines (often a single line).
The chosen length is the same as `black <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#line-length>`_.
