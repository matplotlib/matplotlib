"""
===============
Pyplot With GTK
===============

An example of how to use pyplot to manage your figure windows, but
modify the GUI by accessing the underlying gtk widgets
"""
import matplotlib
matplotlib.use('GTK3Agg')  # or 'GTK3Cairo'
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
plt.plot([1, 2, 3], 'ro-', label='easy as 1 2 3')
plt.plot([1, 4, 9], 'gs--', label='easy as 1 2 3 squared')
plt.legend()


manager = plt.get_current_fig_manager()
# you can also access the window or vbox attributes this way
toolbar = manager.toolbar

# now let's add a button to the toolbar
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
pos = 8  # where to insert this in the mpl toolbar
button = Gtk.Button('Click me')
button.show()


def clicked(button):
    print('hi mom')
button.connect('clicked', clicked)

toolitem = Gtk.ToolItem()
toolitem.show()
toolitem.set_tooltip_text('Click me for fun and profit')

toolitem.add(button)
toolbar.insert(toolitem, pos)
pos += 1

# now let's add a widget to the vbox
label = Gtk.Label()
label.set_markup('Drag mouse over axes for position')
label.show()
vbox = manager.vbox
vbox.pack_start(label, False, False, 0)
vbox.reorder_child(manager.toolbar, -1)


def update(event):
    if event.xdata is None:
        label.set_markup('Drag mouse over axes for position')
    else:
        label.set_markup('<span color="#ef0000">x,y=(%f, %f)</span>' % (event.xdata, event.ydata))

plt.connect('motion_notify_event', update)

plt.show()
