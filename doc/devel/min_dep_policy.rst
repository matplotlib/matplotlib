.. _min_deps_policy:

======================================
Minimum Version of Dependencies Policy
======================================

For the purpose of this document, 'minor version' is in the sense of
SemVer (major, minor, patch) and includes both major and minor
releases. For projects that use date-based versioning, every release
is a 'minor version'.

Matplotlib follows `NEP 29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.

Python and NumPy
================

Matplotlib supports:

- All minor versions of Python released 42 months prior to the
  project, and at minimum the two latest minor versions.
- All minor versions of ``numpy`` released in the 24 months prior
  to the project, and at minimum the last three minor versions.

In ``setup.py``, the ``python_requires`` variable should be set to
the minimum supported version of Python.  All supported minor
versions of Python should be in the test matrix and have binary
artifacts built for the release.

Minimum Python and NumPy version support should be adjusted upward
on every major and minor release, but never on a patch release.

See also the :ref:`list-of-dependency-min-versions`.

Python Dependencies
===================

For Python dependencies we should support at least:

with compiled extensions
  minor versions initially released in the 24 months prior to our planned
  release date or the oldest that support our minimum Python + NumPy

without complied extensions
  minor versions initially released in the 12 months prior to our planned
  release date or the oldest that supports our minimum Python.

We will only bump these dependencies as we need new features or the old
versions no longer support our minimum NumPy or Python.

Test and Documentation Dependencies
===================================

As these packages are only needed for testing or building the docs and
not needed by end-users, we can be more aggressive about dropping
support for old versions.  However, we need to be careful to not
over-run what down-stream packagers support (as most of the run the
tests and build the documentation as part of the packaging process).

We will support at least minor versions of the development
dependencies released in the 12 months prior to our planned release.

We will only bump these as needed or versions no longer support our
minimum Python and numpy.

System and C-dependencies
=========================

For system or C-dependencies (FreeType, GUI frameworks, LaTeX,
Ghostscript, FFmpeg) support as old as practical.  These can be difficult to
install for end-users and we want to be usable on as many systems as
possible.  We will bump these on a case-by-case basis.

.. _list-of-dependency-min-versions:

List of dependency versions
===========================

The following list shows the minimal versions of Python and NumPy dependencies
for different versions of Matplotlib. Follow the links for the full
specification of the dependencies.

==========  ========  ======
Matplotlib  Python    NumPy
==========  ========  ======
3.3         3.6       1.15.0
`3.2`_      3.6       1.11.0
`3.1`_      3.6       1.11.0
`3.0`_      3.5       1.10.0
`2.2`_      2.7, 3.4  1.7.1
`2.1`_      2.7, 3.4  1.7.1
`2.0`_      2.7, 3.4  1.7.1
`1.5`_      2.7, 3.4  1.6
`1.4`_      2.6, 3.3  1.6
`1.3`_      2.6, 3.3  1.5
1.2         2.6, 3.1  1.4
1.1         2.4       1.1
1.0         2.4       1.1
==========  ========  ======

.. _`3.2`: https://matplotlib.org/3.2.0/users/installing.html#dependencies
.. _`3.1`: https://matplotlib.org/3.1.0/users/installing.html#dependencies
.. _`3.0`: https://matplotlib.org/3.0.0/users/installing.html#dependencies
.. _`2.2`: https://matplotlib.org/2.2.0/users/installing.html#dependencies
.. _`2.1`: https://matplotlib.org/2.1.0/users/installing.html#dependencies
.. _`2.0`: https://matplotlib.org/2.0.0/users/installing.html#required-dependencies
.. _`1.5`: https://matplotlib.org/1.5.0/users/installing.html#required-dependencies
.. _`1.4`: https://matplotlib.org/1.4.0/users/installing.html#required-dependencies
.. _`1.3`: https://matplotlib.org/1.3.0/users/installing.html#build-requirements
