import numpy as np
import matplotlib.mlab as mlab


r = mlab.csv2rec('data/aapl.csv')
r.sort()

def daily_return(prices):
    g = np.zeros_like(prices)
    g[1:] = (prices[1:]-prices[:-1])/prices[:-1]
    return g

def volume_code(volume):
    ind = np.searchsorted([1e5,1e6, 5e6,10e6, 1e7], volume)
    return ind

summaryfuncs = (
    ('date', lambda x: [thisdate.year for thisdate in x], 'years'),
    ('date', lambda x: [thisdate.month for thisdate in x], 'months'),
    ('date', lambda x: [thisdate.weekday() for thisdate in x], 'weekday'),
    ('adj_close', daily_return, 'dreturn'),
    ('volume', volume_code, 'volcode'),
    )

rsum = mlab.rec_summarize(r, summaryfuncs)

stats = (
    ('dreturn', len, 'rcnt'),
    ('dreturn', np.mean, 'rmean'),
    ('dreturn', np.median, 'rmedian'),
    ('dreturn', np.std, 'rsigma'),
    )

print 'summary by years'
ry = mlab.rec_groupby(rsum, ('years',), stats)
print mlab. rec2txt(ry)

print 'summary by months'
rm = mlab.rec_groupby(rsum, ('months',), stats)
print mlab.rec2txt(rm)

print 'summary by year and month'
rym = mlab.rec_groupby(rsum, ('years','months'), stats)
print mlab.rec2txt(rym)

print 'summary by volume'
rv = mlab.rec_groupby(rsum, ('volcode',), stats)
print mlab.rec2txt(rv)
