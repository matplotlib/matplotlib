``matplotlib.ticker.EngFormatter`` can computes offsets now
-----------------------------------------------------------

`matplotlib.ticker.EngFormatter` has gained the ability to show an offset text near the
axis. Using logic shared with `matplotlib.ticker.ScalarFormatter`, it is capable of
deciding whether the data qualifies having an offset and show it with an appropriate SI
quantity prefix, and with the supplied ``unit``.

To enable this new behavior, simply pass ``useOffset=True`` when you
instantiate `matplotlib.ticker.EngFormatter`. See example
:doc:`/gallery/ticks/engformatter_offset`.

.. plot:: gallery/ticks/engformatter_offset.py
