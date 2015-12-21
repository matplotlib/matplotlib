#! /usr/bin/python

"""
Test by Karen Tracey for threading problem reported in
http://www.mail-archive.com/matplotlib-devel@lists.sourceforge.net/msg04819.html
and solved by JDH in git commit 175e3ec5bed9144.
"""

from __future__ import print_function
import os
import threading
import traceback

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

thread_count = 8
max_iterations = 50
exception_raised = False


def png_thread(tn):
    png_fname = 'out%d.png' % tn
    vals = 100 + 15 * np.random.randn(10000)

    i = 0
    excp = None
    global exception_raised
    while not exception_raised and i < max_iterations:
        i += 1
        png_f = open(png_fname, 'wb')

        try:
            fig = Figure()
            ax = fig.add_subplot(111)
            ax.hist(vals, 50)
            FigureCanvas(fig).print_png(png_f)

        except Exception as excp:
            pass

        png_f.close()
        if excp:
            print('png_thread %d failed on iteration %d:' % (tn, i))
            print(traceback.format_exc(excp))
            exception_raised = True
        else:
            print('png_thread %d completed iteration %d.' % (tn, i))

    os.unlink(png_fname)


def main(tc):
    threads = []
    for i in range(tc):
        threads.append(threading.Thread(target=png_thread, args=(i + 1,)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    if not exception_raised:
        msg = 'Success! %d threads completed %d iterations with no exceptions raised.'
    else:
        msg = 'Failed! Exception raised before %d threads completed %d iterations.'

    print(msg % (tc, max_iterations))

if __name__ == "__main__":
    main(thread_count)
