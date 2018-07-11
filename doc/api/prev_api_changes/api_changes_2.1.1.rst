API Changes in 2.1.1
====================

Default behavior of log scales reverted to clip <= 0 values
-----------------------------------------------------------

The change it 2.1.0 to mask in logscale by default had more disruptive
changes than anticipated and has been reverted, however the clipping is now
done in a way that fixes the issues that motivated changing the default behavior
to ``'mask'``.

As a side effect of this change, error bars which go negative now work as expected
on log scales.
