import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

def fakeBootStrapper(data):
    '''
    This is just a placeholder for the user's method of
    bootstrapping the median and its confidence intervals.

    Returns an arbitrary median and confidence intervals
    packed into a tuple
    '''
    med, q25, q75 = np.percentile(data, [50,25,75])
    ci_lo = med - (med-q25)/2.0
    ci_hi = med + (q75-med)/3.0
    return med, (ci_lo, ci_hi)

np.random.seed(2)
inc = 0.1
e1 = np.random.normal(0, 1, size=(500,))
e2 = np.random.normal(0, 1, size=(500,))
e3 = np.random.normal(0, 1 + inc, size=(500,))
e4 = np.random.normal(0, 1 + 2*inc, size=(500,))

treatments = [e1,e2,e3,e4]
med1, CI1 = fakeBootStrapper(e3)
med2, CI2 = fakeBootStrapper(e4)
medians = [None, None, med1, med2]
conf_intervals = [None, None, CI1, CI2]

fig, (ax1, ax2) = plt.subplots(nrows=2)
pos = np.array(range(len(treatments)))+1

# specify the medians and CIs manually
bp1 = ax1.boxplot(treatments, sym='k+', positions=pos,
                 notch=1, bootstrap=5000,
                 usermedians=medians,
                 conf_intervals=conf_intervals)

# let the function `fakeBootStrapper` compute the med. and CIs
bp2 = ax2.boxplot(treatments, sym='k+', positions=pos,
                  notch=1, bootstrap=5000,
                  ci_func=fakeBootStrapper)

for ax, bp in zip([ax1, ax2], [bp1, bp2]):
    ax.set_xlabel('treatment')
    ax.set_ylabel('response')
    plt.setp(bp['whiskers'], color='k',  linestyle='-' )
    plt.setp(bp['fliers'], markersize=3.0)
plt.show()
