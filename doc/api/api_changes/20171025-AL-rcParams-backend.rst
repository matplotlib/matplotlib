Testing multiple candidate backends in rcParams
```````````````````````````````````````````````

It is now possible to set ``rcParams["backend"]`` to a *list* of candidate
backends.

If `.pyplot` has already been imported, Matplotlib will try to load each
candidate backend in the given order until one of them can be loaded
successfully. ``rcParams["backend"]`` will then be set to the value of the
successfully loaded backend.  (If `.pyplot` has already been imported and
``rcParams["backend"]`` is set to a single value, then the backend will
likewise be updated.)

If `.pyplot` has not been imported yet, then ``rcParams["backend"]`` will
maintain the value as a list, and the loading attempt will occur when `.pyplot`
is imported.  If you rely on ``rcParams["backend"]`` (or its synonym,
``matplotlib.get_backend()`` always being a string, import `.pyplot` to trigger
backend resolution.

`matplotlib.use`, `pyplot.switch_backends`, and
`matplotlib.backends.pylab_setup` have likewise gained the ability to accept a
list of candidate backends.  Note, however, that the first two functions have
become redundant with directly setting ``rcParams["backend"]``.
