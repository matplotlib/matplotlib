from __future__ import print_function
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook

datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading', datafile)
r = mlab.csv2rec(datafile)
r.sort()


def daily_return(prices):
    'an array of daily returns from price array'
    g = np.zeros_like(prices)
    g[1:] = (prices[1:] - prices[:-1])/prices[:-1]
    return g


def volume_code(volume):
    'code the continuous volume data categorically'
    ind = np.searchsorted([1e5, 1e6, 5e6, 10e6, 1e7], volume)
    return ind

# a list of (dtype_name, summary_function, output_dtype_name).
# rec_summarize will call on each function on the indicated recarray
# attribute, and the result assigned to output name in the return
# record array.
summaryfuncs = (
    ('date', lambda x: [thisdate.year for thisdate in x], 'years'),
    ('date', lambda x: [thisdate.month for thisdate in x], 'months'),
    ('date', lambda x: [thisdate.weekday() for thisdate in x], 'weekday'),
    ('adj_close', daily_return, 'dreturn'),
    ('volume', volume_code, 'volcode'),
    )

rsum = mlab.rec_summarize(r, summaryfuncs)

# stats is a list of (dtype_name, function, output_dtype_name).
# rec_groupby will summarize the attribute identified by the
# dtype_name over the groups in the groupby list, and assign the
# result to the output_dtype_name
stats = (
    ('dreturn', len, 'rcnt'),
    ('dreturn', np.mean, 'rmean'),
    ('dreturn', np.median, 'rmedian'),
    ('dreturn', np.std, 'rsigma'),
    )

# you can summarize over a single variable, like years or months
print('summary by years')
ry = mlab.rec_groupby(rsum, ('years',), stats)
print(mlab. rec2txt(ry))

print('summary by months')
rm = mlab.rec_groupby(rsum, ('months',), stats)
print(mlab.rec2txt(rm))

# or over multiple variables like years and months
print('summary by year and month')
rym = mlab.rec_groupby(rsum, ('years', 'months'), stats)
print(mlab.rec2txt(rym))

print('summary by volume')
rv = mlab.rec_groupby(rsum, ('volcode',), stats)
print(mlab.rec2txt(rv))
