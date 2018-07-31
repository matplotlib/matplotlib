Blacklisted rcparams no longer updated by `rcdefaults`, `rc_file_defaults`, `rc_file`
-------------------------------------------------------------------------------------

The rc modifier functions `rcdefaults`, `rc_file_defaults` and `rc_file`
now ignore rcParams in the `matplotlib.style.core.STYLE_BLACKLIST` set.  In
particular, this prevents the ``backend`` and ``interactive`` rcParams from
being incorrectly modified by these functions.
