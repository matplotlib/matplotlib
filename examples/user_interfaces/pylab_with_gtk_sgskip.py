"""
==============
Pylab With GTK
==============

An example of how to use pylab to manage your figure windows, but
modify the GUI by accessing the underlying gtk widgets
"""
from __future__ import print_function
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
plt.plot([1, 2, 3], 'ro-', label='easy as 1 2 3')
plt.plot([1, 4, 9], 'gs--', label='easy as 1 2 3 squared')
plt.legend()


manager = plt.get_current_fig_manager()
# you can also access the window or vbox attributes this way
toolbar = manager.toolbar

# now let's add a button to the toolbar
import gtk
next = 8  # where to insert this in the mpl toolbar
button = gtk.Button('Click me')
button.show()


def clicked(button):
    print('hi mom')
button.connect('clicked', clicked)

toolitem = gtk.ToolItem()
toolitem.show()
toolitem.set_tooltip(
    toolbar.tooltips,
    'Click me for fun and profit')

toolitem.add(button)
toolbar.insert(toolitem, next)
next += 1

# now let's add a widget to the vbox
label = gtk.Label()
label.set_markup('Drag mouse over axes for position')
label.show()
vbox = manager.vbox
vbox.pack_start(label, False, False)
vbox.reorder_child(manager.toolbar, -1)


def update(event):
    if event.xdata is None:
        label.set_markup('Drag mouse over axes for position')
    else:
        label.set_markup('<span color="#ef0000">x,y=(%f, %f)</span>' % (event.xdata, event.ydata))

plt.connect('motion_notify_event', update)

plt.show()
