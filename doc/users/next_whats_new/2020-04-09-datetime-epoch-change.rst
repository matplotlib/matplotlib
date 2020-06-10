Dates now use a modern epoch
----------------------------

Matplotlib converts dates to days since an epoch using `.dates.date2num` (via
`matplotlib.units`).  Previously, an epoch of ``0000-12-31T00:00:00`` was used
so that ``0001-01-01`` was converted to 1.0.  An epoch so distant in the
past meant that a modern date was not able to preserve microseconds because
2000 years times the 2^(-52) resolution of a 64-bit float gives 14
microseconds.

Here we change the default epoch to the more reasonable UNIX default of
``1970-01-01T00:00:00`` which for a modern date has 0.35 microsecond
resolution.  (Finer resolution is not possible because we rely on
`datetime.datetime` for the date locators). Access to the epoch is provided
by `~.dates.get_epoch`, and there is a new :rc:`date.epoch` rcParam.  The user
may also call `~.dates.set_epoch`, but it must be set *before* any date
conversion or plotting is used.

If you have data stored as ordinal floats in the old epoch, a simple
conversion (using the new epoch) is::

    new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))
