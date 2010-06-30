import datetime
import numpy as np
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt
from nose.tools import assert_raises

@image_comparison(baseline_images=['date_empty'])
def test_date_empty():
    # make sure mpl does the right thing when told to plot dates even
    # if no date data has been presented, cf
    # http://sourceforge.net/tracker/?func=detail&aid=2850075&group_id=80706&atid=560720
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.xaxis_date()
    fig.savefig('date_empty')

@image_comparison(baseline_images=['date_axhspan'])
def test_date_axhspan():
    # test ax hspan with date inputs
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 21)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axhspan( t0, tf, facecolor="blue", alpha=0.25 )
    ax.set_ylim(t0-datetime.timedelta(days=5),
                tf+datetime.timedelta(days=5))
    fig.subplots_adjust(left=0.25)
    fig.savefig('date_axhspan')

@image_comparison(baseline_images=['date_axvspan'])
def test_date_axvspan():
    # test ax hspan with date inputs
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2010, 1, 21)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axvspan( t0, tf, facecolor="blue", alpha=0.25 )
    ax.set_xlim(t0-datetime.timedelta(days=720),
                tf+datetime.timedelta(days=720))
    fig.autofmt_xdate()
    fig.savefig('date_axvspan')


@image_comparison(baseline_images=['date_axhline'])
def test_date_axhline():
    # test ax hline with date inputs
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 31)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axhline( t0, color="blue", lw=3)
    ax.set_ylim(t0-datetime.timedelta(days=5),
                tf+datetime.timedelta(days=5))
    fig.subplots_adjust(left=0.25)
    fig.savefig('date_axhline')

@image_comparison(baseline_images=['date_axvline'])
def test_date_axvline():
    # test ax hline with date inputs
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 21)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axvline( t0, color="red", lw=3)
    ax.set_xlim(t0-datetime.timedelta(days=5),
                tf+datetime.timedelta(days=5))
    fig.autofmt_xdate()
    fig.savefig('date_axvline')

def test_too_many_date_ticks():
    # Attempt to test SF 2715172, see
    # https://sourceforge.net/tracker/?func=detail&aid=2715172&group_id=80706&atid=560720
    # setting equal datetimes triggers and expander call in
    # transforms.nonsingular which results in too many ticks in the
    # DayLocator.  This should trigger a Locator.MAXTICKS RuntimeError
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 20)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim((t0,tf), auto=True)
    ax.plot([],[])
    from matplotlib.dates import DayLocator, DateFormatter, HourLocator
    ax.xaxis.set_major_locator(DayLocator())
    assert_raises(RuntimeError, fig.savefig, 'junk.png')

@image_comparison(baseline_images=['RRuleLocator_bounds'])
def test_RRuleLocator():
    import pylab
    import matplotlib.dates as mpldates
    import matplotlib.testing.jpl_units as units
    from datetime import datetime
    import dateutil
    units.register()

    # This will cause the RRuleLocator to go out of bounds when it tries
    # to add padding to the limits, so we make sure it caps at the correct
    # boundary values.
    t0 = datetime( 1000, 1, 1 )
    tf = datetime( 6000, 1, 1 )

    fig = pylab.figure()
    ax = pylab.subplot( 111 )
    ax.set_autoscale_on( True )
    ax.plot( [t0, tf], [0.0, 1.0], marker='o' )

    rrule = mpldates.rrulewrapper( dateutil.rrule.YEARLY, interval=500 )
    locator = mpldates.RRuleLocator( rrule )
    ax.xaxis.set_major_locator( locator )
    ax.xaxis.set_major_formatter( mpldates.AutoDateFormatter(locator) )

    ax.autoscale_view()
    fig.autofmt_xdate()

    fig.savefig( 'RRuleLocator_bounds' )

@image_comparison(baseline_images=['DateFormatter_fractionalSeconds'])
def test_DateFormatter():
    import pylab
    from datetime import datetime
    import matplotlib.testing.jpl_units as units
    units.register()

    # Lets make sure that DateFormatter will allow us to have tick marks
    # at intervals of fractional seconds.

    t0 = datetime( 2001, 1, 1, 0, 0, 0 )
    tf = datetime( 2001, 1, 1, 0, 0, 1 )

    fig = pylab.figure()
    ax = pylab.subplot( 111 )
    ax.set_autoscale_on( True )
    ax.plot( [t0, tf], [0.0, 1.0], marker='o' )

    # rrule = mpldates.rrulewrapper( dateutil.rrule.YEARLY, interval=500 )
    # locator = mpldates.RRuleLocator( rrule )
    # ax.xaxis.set_major_locator( locator )
    # ax.xaxis.set_major_formatter( mpldates.AutoDateFormatter(locator) )

    ax.autoscale_view()
    fig.autofmt_xdate()

    fig.savefig( 'DateFormatter_fractionalSeconds' )

#@image_comparison(baseline_images=['empty_date_bug'])
@knownfailureif(True)
def test_empty_date_with_year_formatter():
    # exposes sf bug 2861426: https://sourceforge.net/tracker/?func=detail&aid=2861426&group_id=80706&atid=560720

    # update: I am no loner believe this is a bug, as I commented on
    # the tracker.  The question is now: what to do with this test

    import matplotlib.dates as dates

    fig = plt.figure()
    ax = fig.add_subplot(111)

    yearFmt = dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(yearFmt)

    fig.savefig('empty_date_bug')

if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
