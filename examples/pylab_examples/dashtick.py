# Matplotlib xaxis label tweak

import sys
import matplotlib
from matplotlib import pylab, ticker

ROTATION=75
DASHROTATION=75
DASHBASE=10
DASHLEN=35
DASHSTAGGER=3
FONTSIZE=6

def dashlen(step):
    return DASHBASE+(DASHLEN*(step%DASHSTAGGER))

def test_dashticklabel():
    pylab.clf()
    x = [0.0, 1.0, 1.1, 5.0, 5.1, 6.0]
    y = [1, 3, 2, 5, 1, 2]
    labels = ['foo', 'bar', 'baz', 'alpha', 'beta', 'gamma']
    locator = ticker.FixedLocator(x)
    formatter = ticker.FixedFormatter(labels)
    axis = pylab.axes([0.3, 0.3, 0.4, 0.4])
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_locator(locator)
    axis.yaxis.set_major_formatter(formatter)
    for tick in axis.xaxis.get_major_ticks():
        tick.label2On = True
    for tick in axis.yaxis.get_major_ticks():
        tick.label2On = True
    step = 0
    for label in axis.get_xticklabels():
        pylab.setp(label,
                   rotation=ROTATION,
                   dashlength=dashlen(step),
                   dashrotation=DASHROTATION,
                   fontsize=FONTSIZE,
                  )
        step += 1
    step = 0
    for label in axis.get_yticklabels():
        pylab.setp(label,
                   rotation=90-ROTATION,
                   dashlength=dashlen(step),
                   dashrotation=90-DASHROTATION,
                   fontsize=FONTSIZE,
                  )
        step += 1
    pylab.xlabel('X Label')
    pylab.ylabel('Y Label')
    pylab.plot(x, y)
    axis.set_xlim((0.0, 6.0))
    axis.set_ylim((0.0, 6.0))
    pylab.savefig('dashticklabel')
    pylab.show()

if __name__ == '__main__':
    test_dashticklabel()
