``EngFormatter`` significant digits support
--------------------------------------------

The `.ticker.EngFormatter` now supports a ``digits`` parameter to format
tick labels using a fixed number of significant figures across magnitudes,
which is a common requirement in scientific and engineering plots where
values at very different scales should convey comparable precision.

Previously, only the ``places`` parameter was available. That option fixes
the number of decimal places, which can implicitly change the number of
significant figures as the prefix changes. For example, with ``places=1``,
the formatter would produce "10.0 Hz" (3 significant figures) but
"100.0 MHz" (4 significant figures) for similarly precise values.

The new ``digits`` parameter ensures consistent precision regardless of
magnitude:

.. code-block:: python

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    formatter = mticker.EngFormatter(unit='V', digits=4)
    # All values display exactly 4 significant figures
    formatter.format_data(12345)    # "12.35 kV"
    formatter.format_data(123.45)   # "123.5 V"
    formatter.format_data(0.001234) # "1.234 mV"

In addition, a ``trim_zeros`` parameter controls whether trailing zeros are
preserved (``trim_zeros="keep"``, the default, preserving current behavior)
or removed (``trim_zeros="trim"``) for a more compact visual presentation
in tick labels.
