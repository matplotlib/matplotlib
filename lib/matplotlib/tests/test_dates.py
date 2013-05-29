from __future__ import print_function

import datetime
import warnings
import tempfile

from nose.tools import assert_raises, assert_equal
import dateutil

from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DayLocator


@image_comparison(baseline_images=['date_empty'], extensions=['png'])
def test_date_empty():
    # make sure mpl does the right thing when told to plot dates even
    # if no date data has been presented, cf
    # http://sourceforge.net/tracker/?func=detail&aid=2850075&group_id=80706&atid=560720
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis_date()


@image_comparison(baseline_images=['date_axhspan'], extensions=['png'])
def test_date_axhspan():
    # test ax hspan with date inputs
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 21)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axhspan(t0, tf, facecolor="blue", alpha=0.25)
    ax.set_ylim(t0 - datetime.timedelta(days=5),
                tf + datetime.timedelta(days=5))
    fig.subplots_adjust(left=0.25)


@image_comparison(baseline_images=['date_axvspan'], extensions=['png'])
def test_date_axvspan():
    # test ax hspan with date inputs
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2010, 1, 21)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axvspan(t0, tf, facecolor="blue", alpha=0.25)
    ax.set_xlim(t0 - datetime.timedelta(days=720),
                tf + datetime.timedelta(days=720))
    fig.autofmt_xdate()


@image_comparison(baseline_images=['date_axhline'],
                  extensions=['png'])
def test_date_axhline():
    # test ax hline with date inputs
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 31)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(t0, color="blue", lw=3)
    ax.set_ylim(t0 - datetime.timedelta(days=5),
                tf + datetime.timedelta(days=5))
    fig.subplots_adjust(left=0.25)


@image_comparison(baseline_images=['date_axvline'], tol=16,
                  extensions=['png'])
def test_date_axvline():
    # test ax hline with date inputs
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 21)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axvline(t0, color="red", lw=3)
    ax.set_xlim(t0 - datetime.timedelta(days=5),
                tf + datetime.timedelta(days=5))
    fig.autofmt_xdate()


@cleanup
def test_too_many_date_ticks():
    # Attempt to test SF 2715172, see
    # https://sourceforge.net/tracker/?func=detail&aid=2715172&group_id=80706&atid=560720
    # setting equal datetimes triggers and expander call in
    # transforms.nonsingular which results in too many ticks in the
    # DayLocator.  This should trigger a Locator.MAXTICKS RuntimeError
    warnings.filterwarnings(
        'ignore',
        'Attempting to set identical left==right results\\nin singular '
        'transformations; automatically expanding.\\nleft=\d*\.\d*, '
        'right=\d*\.\d*',
        UserWarning, module='matplotlib.axes')
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 20)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((t0, tf), auto=True)
    ax.plot([], [])
    ax.xaxis.set_major_locator(mdates.DayLocator())
    assert_raises(RuntimeError, fig.savefig, 'junk.png')


@image_comparison(baseline_images=['RRuleLocator_bounds'], extensions=['png'])
def test_RRuleLocator():
    import matplotlib.testing.jpl_units as units
    units.register()

    # This will cause the RRuleLocator to go out of bounds when it tries
    # to add padding to the limits, so we make sure it caps at the correct
    # boundary values.
    t0 = datetime.datetime(1000, 1, 1)
    tf = datetime.datetime(6000, 1, 1)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_autoscale_on(True)
    ax.plot([t0, tf], [0.0, 1.0], marker='o')

    rrule = mdates.rrulewrapper(dateutil.rrule.YEARLY, interval=500)
    locator = mdates.RRuleLocator(rrule)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))

    ax.autoscale_view()
    fig.autofmt_xdate()


@image_comparison(baseline_images=['DateFormatter_fractionalSeconds'],
                  extensions=['png'])
def test_DateFormatter():
    import matplotlib.testing.jpl_units as units
    units.register()

    # Lets make sure that DateFormatter will allow us to have tick marks
    # at intervals of fractional seconds.

    t0 = datetime.datetime(2001, 1, 1, 0, 0, 0)
    tf = datetime.datetime(2001, 1, 1, 0, 0, 1)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_autoscale_on(True)
    ax.plot([t0, tf], [0.0, 1.0], marker='o')

    # rrule = mpldates.rrulewrapper( dateutil.rrule.YEARLY, interval=500 )
    # locator = mpldates.RRuleLocator( rrule )
    # ax.xaxis.set_major_locator( locator )
    # ax.xaxis.set_major_formatter( mpldates.AutoDateFormatter(locator) )

    ax.autoscale_view()
    fig.autofmt_xdate()


def test_drange():
    """
    This test should check if drange works as expected, and if all the
    rounding errors are fixed
    """
    start = datetime.datetime(2011, 1, 1, tzinfo=mdates.UTC)
    end = datetime.datetime(2011, 1, 2, tzinfo=mdates.UTC)
    delta = datetime.timedelta(hours=1)
    # We expect 24 values in drange(start, end, delta), because drange returns
    # dates from an half open interval [start, end)
    assert_equal(24, len(mdates.drange(start, end, delta)))

    # if end is a little bit later, we expect the range to contain one element
    # more
    end = end + datetime.timedelta(microseconds=1)
    assert_equal(25, len(mdates.drange(start, end, delta)))

    # reset end
    end = datetime.datetime(2011, 1, 2, tzinfo=mdates.UTC)

    # and tst drange with "complicated" floats:
    # 4 hours = 1/6 day, this is an "dangerous" float
    delta = datetime.timedelta(hours=4)
    daterange = mdates.drange(start, end, delta)
    assert_equal(6, len(daterange))
    assert_equal(mdates.num2date(daterange[-1]), end - delta)


@cleanup
def test_empty_date_with_year_formatter():
    # exposes sf bug 2861426:
    # https://sourceforge.net/tracker/?func=detail&aid=2861426&group_id=80706&atid=560720

    # update: I am no longer believe this is a bug, as I commented on
    # the tracker.  The question is now: what to do with this test

    import matplotlib.dates as dates

    fig = plt.figure()
    ax = fig.add_subplot(111)

    yearFmt = dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(yearFmt)

    with tempfile.TemporaryFile() as fh:
        assert_raises(ValueError, fig.savefig, fh)


def test_auto_date_locator():
    def _create_auto_date_locator(date1, date2):
        locator = mdates.AutoDateLocator()
        locator.create_dummy_axis()
        locator.set_view_interval(mdates.date2num(date1), mdates.date2num(date2))
        return locator

    d1 = datetime.datetime(1990, 1, 1)
    results = ([datetime.timedelta(weeks=52 * 200),
                ['1990-01-01 00:00:00+00:00', '2010-01-01 00:00:00+00:00',
                 '2030-01-01 00:00:00+00:00', '2050-01-01 00:00:00+00:00',
                 '2070-01-01 00:00:00+00:00', '2090-01-01 00:00:00+00:00',
                 '2110-01-01 00:00:00+00:00', '2130-01-01 00:00:00+00:00',
                 '2150-01-01 00:00:00+00:00', '2170-01-01 00:00:00+00:00']
                ],
               [datetime.timedelta(weeks=52),
                ['1990-01-01 00:00:00+00:00', '1990-02-01 00:00:00+00:00',
                 '1990-03-01 00:00:00+00:00', '1990-04-01 00:00:00+00:00',
                 '1990-05-01 00:00:00+00:00', '1990-06-01 00:00:00+00:00',
                 '1990-07-01 00:00:00+00:00', '1990-08-01 00:00:00+00:00',
                 '1990-09-01 00:00:00+00:00', '1990-10-01 00:00:00+00:00',
                 '1990-11-01 00:00:00+00:00', '1990-12-01 00:00:00+00:00']
                ],
               [datetime.timedelta(days=140),
                ['1990-01-06 00:00:00+00:00', '1990-01-27 00:00:00+00:00',
                 '1990-02-17 00:00:00+00:00', '1990-03-10 00:00:00+00:00',
                 '1990-03-31 00:00:00+00:00', '1990-04-21 00:00:00+00:00',
                 '1990-05-12 00:00:00+00:00']
                ],
               [datetime.timedelta(days=40),
                ['1990-01-03 00:00:00+00:00', '1990-01-10 00:00:00+00:00',
                 '1990-01-17 00:00:00+00:00', '1990-01-24 00:00:00+00:00',
                 '1990-01-31 00:00:00+00:00', '1990-02-07 00:00:00+00:00']
                ],
               [datetime.timedelta(hours=40),
                ['1990-01-01 00:00:00+00:00', '1990-01-01 04:00:00+00:00',
                 '1990-01-01 08:00:00+00:00', '1990-01-01 12:00:00+00:00',
                 '1990-01-01 16:00:00+00:00', '1990-01-01 20:00:00+00:00',
                 '1990-01-02 00:00:00+00:00', '1990-01-02 04:00:00+00:00',
                 '1990-01-02 08:00:00+00:00', '1990-01-02 12:00:00+00:00',
                 '1990-01-02 16:00:00+00:00']
                ],
               [datetime.timedelta(minutes=20),
                ['1990-01-01 00:00:00+00:00', '1990-01-01 00:05:00+00:00',
                 '1990-01-01 00:10:00+00:00', '1990-01-01 00:15:00+00:00',
                 '1990-01-01 00:20:00+00:00']

                ],
               [datetime.timedelta(seconds=40),
                ['1990-01-01 00:00:00+00:00', '1990-01-01 00:00:05+00:00',
                 '1990-01-01 00:00:10+00:00', '1990-01-01 00:00:15+00:00',
                 '1990-01-01 00:00:20+00:00', '1990-01-01 00:00:25+00:00',
                 '1990-01-01 00:00:30+00:00', '1990-01-01 00:00:35+00:00',
                 '1990-01-01 00:00:40+00:00']
                ],
               [datetime.timedelta(microseconds=1500),
                ['1989-12-31 23:59:59.999507+00:00', '1990-01-01 00:00:00+00:00',
                 '1990-01-01 00:00:00.000502+00:00', '1990-01-01 00:00:00.001005+00:00',
                 '1990-01-01 00:00:00.001508+00:00']
                ],
               )

    for t_delta, expected in results:
        d2 = d1 + t_delta
        locator = _create_auto_date_locator(d1, d2)
        assert_equal(map(str, mdates.num2date(locator())),
                     expected)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
