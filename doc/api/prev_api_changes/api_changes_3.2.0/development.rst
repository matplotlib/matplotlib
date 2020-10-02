Development changes
-------------------

Windows build
~~~~~~~~~~~~~
Previously, when building the ``matplotlib._png`` extension, the build
script would add "png" and "z" to the extensions ``.libraries`` attribute (if
pkg-config information is not available, which is in particular the case on
Windows).

In particular, this implies that the Windows build would look up files named
``png.lib`` and ``z.lib``; but neither libpng upstream nor zlib upstream
provides these files by default.  (On Linux, this would look up ``libpng.so``
and ``libz.so``, which are indeed standard names.)

Instead, on Windows, we now look up ``libpng16.lib`` and ``zlib.lib``, which
*are* the upstream names for the shared libraries (as of libpng 1.6.x).

For a statically-linked build, the upstream names are ``libpng16_static.lib``
and ``zlibstatic.lib``; one still needs to manually rename them if such a build
is desired.

Packaging DLLs
~~~~~~~~~~~~~~
Previously, it was possible to package Windows DLLs into the Maptlotlib
wheel (or sdist) by copying them into the source tree and setting the
``package_data.dlls`` entry in ``setup.cfg``.

DLLs copied in the source tree are now always packaged; the
``package_data.dlls`` entry has no effect anymore.  If you do not want to
include the DLLs, don't copy them into the source tree.
