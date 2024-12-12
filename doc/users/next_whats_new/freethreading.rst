Preliminary support for free-threaded CPython 3.13
--------------------------------------------------

Matplotlib 3.10 has preliminary support for the free-threaded build of CPython 3.13. See
https://py-free-threading.github.io, `PEP 703 <https://peps.python.org/pep-0703/>`_ and
the `CPython 3.13 release notes
<https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython>`_ for more detail
about free-threaded Python.

Support for free-threaded Python does not mean that Matplotlib is wholly thread safe. We
expect that use of a Figure within a single thread will work, and though input data is
usually copied, modification of data objects used for a plot from another thread may
cause inconsistencies in cases where it is not. Use of any global state (such as the
``pyplot`` module) is highly discouraged and unlikely to work consistently. Also note
that most GUI toolkits expect to run on the main thread, so interactive usage may be
limited or unsupported from other threads.

If you are interested in free-threaded Python, for example because you have a
multiprocessing-based workflow that you are interested in running with Python threads, we
encourage testing and experimentation. If you run into problems that you suspect are
because of Matplotlib, please open an issue, checking first if the bug also occurs in the
“regular” non-free-threaded CPython 3.13 build.
