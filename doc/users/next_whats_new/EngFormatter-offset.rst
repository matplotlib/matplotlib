ticker.EngFormatter now computes offset by default
--------------------------------------------------

``ticker.EngFormatter`` was modified to act very similar to
``ticker.ScalarFormatter``, such that it computes the best offset of the axis
data, and shows the offset with the known SI quantity prefixes. To disable this
new behavior, simply pass ``useOffset=False`` when you instantiate it. If offsets
are disabled, or if there is no particular offset that fits your axis data, the
formatter will reside to the old behavior.
