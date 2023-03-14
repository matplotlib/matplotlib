Changes of API after deprecation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `.dviread.find_tex_file` now raises `FileNotFoundError` when the requested filename is
  not found.
- `.Figure.colorbar` now raises if *cax* is not given and it is unable to determine from
  which Axes to steal space, i.e. if *ax* is also not given and *mappable* has not been
  added to an Axes.
- `.pyplot.subplot` and `.pyplot.subplot2grid` no longer auto-remove preexisting
  overlapping Axes; explicitly call ``Axes.remove`` as needed.
