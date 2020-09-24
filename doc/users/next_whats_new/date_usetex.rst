Date formatters now respect *usetex* rcParam
--------------------------------------------

The `.AutoDateFormatter` and `.ConciseDateFormatter` now respect
:rc:`text.usetex`, and will thus use fonts consistent with TeX rendering of the
default (non-date) formatter. TeX rendering may also be enabled/disabled by
passing the *usetex* parameter when creating the formatter instance.

In the following plot, both the x-axis (dates) and y-axis (numbers) now use the
same (TeX) font:

.. plot::

    from datetime import datetime, timedelta
    from matplotlib.dates import ConciseDateFormatter

    plt.rc('text', usetex=True)

    t0 = datetime(1968, 8, 1)
    ts = [t0 + i * timedelta(days=1) for i in range(10)]

    fig, ax = plt.subplots()
    ax.plot(ts, range(10))
    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
