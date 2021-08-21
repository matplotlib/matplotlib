Matplotlib-specific build options have moved from ``setup.cfg`` to ``mplsetup.cfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... in order to avoid conflicting with the use of :file:`setup.cfg` by
``setuptools``.

Note that the path to this configuration file can still be set via the
:envvar:`MPLSETUPCFG` environment variable, which allows one to keep using the
same file before and after this change.
