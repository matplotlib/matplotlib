import datetime
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

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

@image_comparison(baseline_images=['date_xlim_empty'])
def test_set_xlim_and_unexpected_handling():
    # Attempt to test SF 2715172, see
    # https://sourceforge.net/tracker/?func=detail&aid=2715172&group_id=80706&atid=560720
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 20)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim((t0,tf))
    ax.plot([],[])
    from matplotlib.dates import DayLocator, DateFormatter, HourLocator
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d/%y, %I:%M%p"))
    ax.xaxis.set_minor_locator(HourLocator())
    if 1:
        # this seems to cause an ininite loop.
        from nose.plugins.skip import SkipTest
        raise SkipTest('avoiding never-ending drawing')
    fig.savefig('date_xlim_empty')

if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
