from __future__ import print_function
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook

datafile = cbook.get_sample_data('msft.csv', asfileobj=False)
print('loading', datafile)
r = mlab.csv2rec(datafile)

r.sort()
r1 = r[-10:]

# Create a new array
r2 = np.empty(12, dtype=[('date', '|O4'), ('high', float), ('marker', float)])
r2 = r2.view(np.recarray)
r2.date = r.date[-17:-5]
r2.high = r.high[-17:-5]
r2.marker = np.arange(12)

print("r1:")
print(mlab.rec2txt(r1))
print("r2:")
print(mlab.rec2txt(r2))

defaults = {'marker': -1, 'close': np.NaN, 'low': -4444.}

for s in ('inner', 'outer', 'leftouter'):
    rec = mlab.rec_join(['date', 'high'], r1, r2,
                        jointype=s, defaults=defaults)
    print("\n%sjoin :\n%s" % (s, mlab.rec2txt(rec)))
