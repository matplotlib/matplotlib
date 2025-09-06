import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


def violin_plot_stats():
    # Stats for violin plot
    datetimes = [
        datetime.datetime(2023, 2, 10),
        datetime.datetime(2023, 5, 18),
        datetime.datetime(2023, 6, 6)
    ]
    return [{
        'coords': datetimes,
        'vals': [0.1, 0.5, 0.2],
        'mean': datetimes[1],
        'median': datetimes[1],
        'min': datetimes[0],
        'max': datetimes[-1],
        'quantiles': datetimes
    }, {
        'coords': datetimes,
        'vals': [0.2, 0.3, 0.4],
        'mean': datetimes[2],
        'median': datetimes[2],
        'min': datetimes[0],
        'max': datetimes[-1],
        'quantiles': datetimes
    }]


def test_violin_datetime_positions_timedelta_widths():
    fig, ax = plt.subplots()
    vpstats = violin_plot_stats()
    positions = [datetime.datetime(2020, 1, 1), datetime.datetime(2021, 1, 1)]
    widths = [datetime.timedelta(days=10), datetime.timedelta(days=20)]
    ax.violin(vpstats, positions=positions, widths=widths)
    plt.close(fig)


def test_violin_date_positions_float_widths():
    fig, ax = plt.subplots()
    vpstats = violin_plot_stats()
    positions = [datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)]
    widths = [0.5, 1.0]
    ax.violin(vpstats, positions=positions, widths=widths)
    plt.close(fig)


def test_violin_mixed_positions_widths():
    fig, ax = plt.subplots()
    vpstats = violin_plot_stats()
    positions = [datetime.datetime(2020, 1, 1),
                 mdates.date2num(datetime.datetime(2021, 1, 1))]
    widths = [datetime.timedelta(days=3), 2.0]
    ax.violin(vpstats, positions=positions, widths=widths)
    plt.close(fig)


def test_violin_default_positions_widths():
    fig, ax = plt.subplots()
    vpstats = violin_plot_stats()
    ax.violin(vpstats)
    plt.close(fig)
