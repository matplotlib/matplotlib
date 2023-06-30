Wheels for some systems are no longer distributed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-compiled wheels for 32-bit Linux and Windows are no longer provided on PyPI
since Matplotlib 3.8.

Multi-architecture ``universal2`` wheels for macOS are no longer provided on PyPI since
Matplotlib 3.8. In general, ``pip`` will always prefer the architecture-specific
(``amd64``- or ``arm64``-only) wheels, so these provided little benefit.
