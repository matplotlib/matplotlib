``EngFormatter`` now respects ``axes.formatter.use_locale``
-----------------------------------------------------------

`~matplotlib.ticker.EngFormatter` now formats the decimal separator according to
the current locale, like `~matplotlib.ticker.ScalarFormatter` already did. This
is controlled by the :rc:`axes.formatter.use_locale` rcParam, or by the new
*useLocale* keyword argument:

.. code-block:: python

    import locale
    from matplotlib.ticker import EngFormatter

    locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")

    ax.yaxis.set_major_formatter(EngFormatter(places=2, useLocale=True))
    # tick labels now read "555,10 m" instead of "555.10 m"
