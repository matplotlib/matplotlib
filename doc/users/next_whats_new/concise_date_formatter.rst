:orphan:

New date formatter: `~.dates.ConciseDateFormatter`
--------------------------------------------------

The automatic date formatter used by default can be quite verbose.  A new
formatter can be accessed that tries to make the tick labels appropriately
concise.

  .. plot::

    import datetime
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # make a timeseries...
    base = datetime.datetime(2005, 2, 1)
    dates = np.array([base + datetime.timedelta(hours= 2 * i)
                      for i in range(732)])
    N = len(dates)
    np.random.seed(19680801)
    y = np.cumsum(np.random.randn(N))

    lims = [(np.datetime64('2005-02'), np.datetime64('2005-04')),
            (np.datetime64('2005-02-03'), np.datetime64('2005-02-15')),
            (np.datetime64('2005-02-03 11:00'), np.datetime64('2005-02-04 13:20'))]
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    for nn, ax in enumerate(axs):
        # activate the formatter here.
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.plot(dates, y)
        ax.set_xlim(lims[nn])
    axs[0].set_title('Concise Date Formatter')

    plt.show()
