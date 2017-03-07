Invalid (Non-finite) Axis Limit Error
-------------------------------------

When using :func:`set_xlim` and :func:`set_ylim`, passing non-finite values now
results in a ValueError. The previous behavior resulted in the limits being
erroneously reset to `(-0.001, 0.001)`.
