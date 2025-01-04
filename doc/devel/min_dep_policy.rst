.. _min_deps_policy:

=========================
Dependency version policy
=========================

For the purpose of this document, 'minor version' is in the sense of SemVer
(major, minor, patch) or 'meso version' in the sense of `EffVer
<https://jacobtomlinson.dev/effver/>`_ (macro, meso, micro).  It includes both
major/macro and minor/meso releases.  For projects that use date-based
versioning, every release is a 'minor version'.

Matplotlib follows `NEP 29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.

Python and NumPy
================

Matplotlib supports:

- All minor versions of Python released 42 months prior to the
  project, and at minimum the two latest minor versions.
- All minor versions of ``numpy`` released in the 24 months prior
  to the project, and at minimum the last three minor versions.

In :file:`pyproject.toml`, the ``requires-python`` variable should be set to
the minimum supported version of Python.  All supported minor
versions of Python should be in the test matrix and have binary
artifacts built for the release.

Minimum Python and NumPy version support should be adjusted upward
on every major and minor release, but never on a patch release.

See also the :ref:`list-of-dependency-min-versions`.

Python dependencies
===================

For Python dependencies we should support at least:

with compiled extensions
  minor versions initially released in the 24 months prior to our planned
  release date or the oldest that support our minimum Python + NumPy

without compiled extensions
  minor versions initially released in the 12 months prior to our planned
  release date or the oldest that supports our minimum Python.

We will only bump these dependencies as we need new features or the old
versions no longer support our minimum NumPy or Python.

We will work around bugs in our dependencies when practical.

IPython and Matplotlib do not formally depend on each other, however there is
practical coupling for the integration of Matplotlib's UI into IPython and
IPykernel.  We will ensure this integration works with at least minor or major
versions of IPython and IPykernel released in the 24 months prior to our
planned release date.  Matplotlib may or may not work with older versions and
we will not warn if used with IPython or IPykernel outside of this window.



Test and documentation dependencies
===================================

As these packages are only needed for testing or building the docs and
not needed by end-users, we can be more aggressive about dropping
support for old versions.  However, we need to be careful to not
over-run what down-stream packagers support (as most of the run the
tests and build the documentation as part of the packaging process).

We will support at least minor versions of the development dependencies
released in the 12 months prior to our planned release.  Specific versions that
are known to be buggy may be excluded from support using the finest-grained
filtering that is practical.

We will only bump these as needed or versions no longer support our
minimum Python and NumPy.

System and C-dependencies
=========================

For system or C-dependencies (FreeType, GUI frameworks, LaTeX,
Ghostscript, FFmpeg) support as old as practical.  These can be difficult to
install for end-users and we want to be usable on as many systems as
possible.  We will bump these on a case-by-case basis.

In the case of GUI frameworks for which we rely on Python bindings being
available, we will also drop support for bindings so old that they don't
support any Python version that we support.

Security issues in dependencies
===============================

Generally, we do not adjust the supported versions of dependencies based on
security vulnerabilities.   We are a library not an application
and the version constraints on our dependencies indicate what will work (not
what is wise to use).  Users and packagers can install newer versions of the
dependencies at their discretion and evaluation of risk and impact.  In
contrast, if we were to adjust our minimum supported version it is very hard
for a user to override our judgment.

If Matplotlib aids in exploiting the underlying vulnerability we should treat
that as a critical bug in Matplotlib.

.. _list-of-dependency-min-versions:

List of dependency versions
===========================

The following list shows the minimal versions of Python and NumPy dependencies
for different versions of Matplotlib. Follow the links for the full
specification of the dependencies.

==========  ========  ======
Matplotlib  Python    NumPy
==========  ========  ======
`3.10`_     3.10      1.23.0
`3.9`_      3.9       1.23.0
`3.8`_      3.9       1.21.0
`3.7`_      3.8       1.20.0
`3.6`_      3.8       1.19.0
`3.5`_      3.7       1.17.0
`3.4`_      3.7       1.16.0
`3.3`_      3.6       1.15.0
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

.. _`3.10`: https://matplotlib.org/3.10.0/devel/dependencies.html
.. _`3.9`: https://matplotlib.org/3.9.0/devel/dependencies.html
.. _`3.8`: https://matplotlib.org/3.8.0/devel/dependencies.html
.. _`3.7`: https://matplotlib.org/3.7.0/devel/dependencies.html
.. _`3.6`: https://matplotlib.org/3.6.0/devel/dependencies.html
.. _`3.5`: https://matplotlib.org/3.5.0/devel/dependencies.html
.. _`3.4`: https://matplotlib.org/3.4.0/devel/dependencies.html
.. _`3.3`: https://matplotlib.org/3.3.0/users/installing.html#dependencies
.. _`3.2`: https://matplotlib.org/3.2.0/users/installing.html#dependencies
.. _`3.1`: https://matplotlib.org/3.1.0/users/installing.html#dependencies
.. _`3.0`: https://matplotlib.org/3.0.0/users/installing.html#dependencies
.. _`2.2`: https://matplotlib.org/2.2.0/users/installing.html#dependencies
.. _`2.1`: https://matplotlib.org/2.1.0/users/installing.html#dependencies
.. _`2.0`: https://matplotlib.org/2.0.0/users/installing.html#required-dependencies
.. _`1.5`: https://matplotlib.org/1.5.0/users/installing.html#required-dependencies
.. _`1.4`: https://matplotlib.org/1.4.0/users/installing.html#required-dependencies
.. _`1.3`: https://matplotlib.org/1.3.0/users/installing.html#build-requirements
