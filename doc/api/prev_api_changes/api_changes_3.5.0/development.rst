Development changes
-------------------

Increase to minimum supported versions of dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Matplotlib 3.5, the :ref:`minimum supported versions <dependencies>` and
some :ref:`optional dependencies <optional_dependencies>` are being bumped:

+---------------+---------------+---------------+
| Dependency    | min in mpl3.4 | min in mpl3.5 |
+===============+===============+===============+
| NumPy         |     1.16      |     1.17      |
+---------------+---------------+---------------+
| Tk (optional) |     8.3       |     8.4       |
+---------------+---------------+---------------+

This is consistent with our :ref:`min_deps_policy` and `NEP29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`__

New wheel architectures
~~~~~~~~~~~~~~~~~~~~~~~

Wheels have been added for:

- Python 3.10
- PyPy 3.7
- macOS on Apple Silicon (both arm64 and universal2)

New build dependencies
~~~~~~~~~~~~~~~~~~~~~~

Versioning has been switched from bundled versioneer to `setuptools-scm
<https://github.com/pypa/setuptools_scm/>`__ using the
``release-branch-semver`` version scheme. The latter is well-maintained, but
may require slight modification to packaging scripts.

The `setuptools-scm-git-archive
<https://pypi.org/project/setuptools-scm-git-archive/>`__ plugin is also used
for consistent version export.

Data directory is no longer optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Historically, the ``mpl-data`` directory has been optional (example files were
unnecessary, and fonts could be deleted if a suitable dependency on a system
font were provided). Though example files are still optional, they have been
substantially pared down, and we now consider the directory to be required.

Specifically, the ``matplotlibrc`` file found there is used for runtime
verifications and must exist. Packagers may still symlink fonts to system
versions if needed.

New runtime dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

fontTools for type 42 subsetting
................................

A new dependency `fontTools <https://fonttools.readthedocs.io/>`_ is integrated
into Matplotlib 3.5. It is designed to be used with PS/EPS and PDF documents;
and handles Type 42 font subsetting.

Underscore support in LaTeX
...........................

The `underscore <https://ctan.org/pkg/underscore>`_ package is now a
requirement to improve support for underscores in LaTeX.

This is consistent with our :ref:`min_deps_policy`.

Matplotlib-specific build options moved from ``setup.cfg`` to ``mplsetup.cfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to avoid conflicting with the use of :file:`setup.cfg` by
``setuptools``, the Matplotlib-specific build options have moved from
``setup.cfg`` to ``mplsetup.cfg``

Note that the path to this configuration file can still be set via the
:envvar:`MPLSETUPCFG` environment variable, which allows one to keep using the
same file before and after this change.
