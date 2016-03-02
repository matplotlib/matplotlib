"""
generate an editable gtk treeview widget for record arrays with custom
formatting of the cells and show how to limit string entries to a list
of strings
"""
from __future__ import print_function
import gtk
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import mpl_toolkits.gtktools as gtktools


datafile = cbook.get_sample_data('demodata.csv', asfileobj=False)
r = mlab.csv2rec(datafile, converterd={'weekdays': str})


formatd = mlab.get_formatd(r)
formatd['date'] = mlab.FormatDate('%Y-%m-%d')
formatd['prices'] = mlab.FormatMillions(precision=1)
formatd['gain'] = mlab.FormatPercent(precision=2)

# use a drop down combo for weekdays
stringd = dict(weekdays=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
constant = ['clientid']   # block editing of this field


liststore = gtktools.RecListStore(r, formatd=formatd, stringd=stringd)
treeview = gtktools.RecTreeView(liststore, constant=constant)


def mycallback(liststore, rownum, colname, oldval, newval):
    print('verify: old=%s, new=%s, rec=%s' % (oldval, newval, liststore.r[rownum][colname]))

liststore.callbacks.connect('cell_changed', mycallback)

win = gtk.Window()
win.set_title('click to edit')
win.add(treeview)
win.show_all()
win.connect('delete-event', lambda *args: gtk.main_quit())
gtk.main()
