TeX support cache
-----------------

The `usetex` feature sends snippets of TeX code to LaTeX and related
external tools for processing. This causes a nontrivial number of
helper processes to be spawned, which can be slow on some platforms.
A new cache database helps reduce the need to spawn these helper
processes, which should improve `usetex` processing speed.

The new cache files
~~~~~~~~~~~~~~~~~~~

The cache database is stored in a file named `texsupport.N.db` in the
standard cache directory (traditionally `$HOME/.matplotlib` but
possibly `$HOME/.cache/matplotlib`), where `N` stands for a version
number. The version number is incremented when new kinds of items are
added to the caching code, in order to avoid version clashes when
using multiple different versions of Matplotlib. The auxiliary files
`texsupport.N.db-wal` and `texsupport.N.db-shm` help coordinate usage
of the cache between concurrently running instances. All of these
cache files may be deleted when Matplotlib is not running, and
subsequent calls to the `usetex` code will recompute the TeX results.
