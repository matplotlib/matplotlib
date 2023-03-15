*np_load* parameter of ``cbook.get_sample_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This parameter is deprecated; `.get_sample_data` now auto-loads numpy arrays.
Use ``get_sample_data(..., asfileobj=False)`` instead to get the filename of
the data file, which can then be passed to `open`, if desired.
