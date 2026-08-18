New sphinx role ``:rcdefault:``
------------------------------

The ``:rc:`` role no longer renders the default value of the referenced
``rcParams`` entry. Use the new ``:rcdefault:`` role to also show the default
value, e.g. ``:rcdefault:`figure.dpi` `` renders as ``rcParams["figure.dpi"]
(default: 100.0)``.
