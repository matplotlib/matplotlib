Development changes
-------------------

Build system ported to Meson
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The build system of Matplotlib has been ported from setuptools to `meson-python
<https://meson-python.readthedocs.io>`_ and `Meson <https://mesonbuild.com>`_.
Consequently, there have been a few changes for development and packaging purposes.

1. Installation by ``pip`` of packages with ``pyproject.toml`` use `build isolation
   <https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/#build-isolation>`_
   by default, which interferes with editable installation. Thus for developers using
   editable installs, it is now necessary to pass the ``--no-build-isolation`` flag to
   ``pip install``. This means that all build-time requirements must be available in the
   environment for an editable install.
2. Build configuration has moved from a custom :file:`mplsetup.cfg` (also configurable
   via ``MPLSETUP`` environment variable) to Meson options. These may be specified using
   `meson-python's build config settings
   <https://meson-python.readthedocs.io/en/stable/how-to-guides/config-settings.html>`_
   for ``setup-args``. See :file:`meson_options.txt` for all options. For example, a
   :file:`mplsetup.cfg` containing the following::

      [rc_options]
      backend=Agg

      [libs]
      system_qhull = True

   may be replaced by passing the following arguments to ``pip``::

      --config-settings=setup-args="-DrcParams-backend=Agg"
      --config-settings=setup-args="-Dsystem-qhull=true"

   Note that you must use ``pip`` >= 23.1 in order to pass more than one setting.
3. Relatedly, Meson's `builtin options <https://mesonbuild.com/Builtin-options.html>`_
   are now used instead of custom options, e.g., the LTO option is now ``b_lto``.
4. On Windows, Meson activates a Visual Studio environment automatically. However, it
   will not do so if another compiler is available. See `Meson's documentation
   <https://mesonbuild.com/Builtin-options.html#details-for-vsenv>`_ if you wish to
   change the priority of chosen compilers.
5. Installation of test data was previously controlled by :file:`mplsetup.cfg`, but has
   now been moved to Meson's install tags. To install test data, add the ``tests`` tag
   to the requested install (be sure to include the existing tags as below)::

      --config-settings=install-args="--tags=data,python-runtime,runtime,tests"
6. Checking typing stubs with ``stubtest`` does not work easily with editable install.
   For the time being, we suggest using a normal (non-editable) install if you wish to
   run ``stubtest``.

Increase to minimum supported versions of dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Matplotlib 3.9, the :ref:`minimum supported versions <dependencies>` are being
bumped:

+------------+-----------------+---------------+
| Dependency |  min in mpl3.8  | min in mpl3.9 |
+============+=================+===============+
|   NumPy    |       1.21.0    |      1.23.0   |
+------------+-----------------+---------------+
| setuptools |       42        |      64       |
+------------+-----------------+---------------+

This is consistent with our :ref:`min_deps_policy` and `SPEC 0
<https://scientific-python.org/specs/spec-0000/>`__.

To comply with requirements of ``setuptools_scm``, the minimum version of ``setuptools``
has been increased from 42 to 64.

Extensions require C++17
^^^^^^^^^^^^^^^^^^^^^^^^

Matplotlib now requires a compiler that supports C++17 in order to build its extensions.
According to `SciPy's analysis
<https://docs.scipy.org/doc/scipy/dev/toolchain.html#c-language-standards>`_, this
should be available on all supported platforms.

Windows on ARM64 support
^^^^^^^^^^^^^^^^^^^^^^^^

Windows on ARM64 now bundles FreeType 2.6.1 instead of 2.11.1 when building from source.
This may cause small changes to text rendering, but should become consistent with all
other platforms.
