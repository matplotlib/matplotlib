"""
Load a CSV file into a record array and edit it in a gtk treeview
"""

import gtk
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import mpl_toolkits.gtktools as gtktools

datafile = cbook.get_sample_data('demodata.csv', asfileobj=False)
r = mlab.csv2rec(datafile, converterd={'weekdays': str})

liststore, treeview, win = gtktools.edit_recarray(r)
win.set_title('click to edit')
win.connect('delete-event', lambda *args: gtk.main_quit())
gtk.main()
