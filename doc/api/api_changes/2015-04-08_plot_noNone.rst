Disallow ``None`` as x or y value in ax.plot
````````````````````````````````````````````

Do not allow ``None`` as a valid input for the ``x`` or ``y`` args in
`ax.plot`.  This may break some user code, but this was never officially
supported (ex documented) and allowing ``None`` objects through can lead
to confusing exceptions downstream.

To create an empty line use ::

  ln1, = ax.plot([], [], ...)
  ln2, = ax.plot([], ...)

In either case to update the data in the `Line2D` object you must update
both the ``x`` and ``y`` data.
