import matplotlib

matplotlib.verbose.report_error("""\
matplotlib.matlab deprecated, please import matplotlib.pylab or simply
pylab instead.  See http://matplotlib.sf.net/matplotlib_to_pylab.py
for a script which explains this change and will automatically convert
your python scripts that use matplotlib.matlab.  This change was made
because we were concerned about trademark infringement on The
Mathwork's trademark of matlab.
""")

from matplotlib.pylab import *
