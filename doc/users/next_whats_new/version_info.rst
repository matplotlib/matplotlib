Version information
-------------------
We switched to the `release-branch-semver`_ version scheme. This only affects,
the version information for development builds. Their version number now
describes the targeted release, i.e. 3.5.0.dev820+g6768ef8c4c.d20210520
is 820 commits after the previous release and is scheduled to be officially
released as 3.5.0 later.

In addition to the string ``__version__``, there is now a namedtuple
``__version_info__`` as well, which is modelled after `sys.version_info`_.
Its primary use is safely comparing version information, e.g.
``if __version_info__ >= (3, 4, 2)``.

.. _release-branch-semver: https://github.com/pypa/setuptools_scm#version-number-construction
.. _sys.version_info: https://docs.python.org/3/library/sys.html#sys.version_info