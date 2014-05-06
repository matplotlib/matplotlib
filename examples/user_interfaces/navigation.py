import matplotlib
matplotlib.use('GTK3Cairo')
# matplotlib.use('TkAGG')
matplotlib.rcParams['toolbar'] = 'navigation'
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase


# Create a simple tool to list all the tools
class ListTools(ToolBase):
    # keyboard shortcut
    keymap = 'm'
    # Name used as id, must be unique between tools of the same navigation
    name = 'List'
    description = 'List Tools'
    # Where to put it in the toolbar, -1 = at the end, None = Not in toolbar
    position = -1

    def trigger(self, event):
        # The most important attributes are navigation and figure
        self.navigation.list_tools()


# A simple example of copy canvas
# ref: at https://github.com/matplotlib/matplotlib/issues/1987
class CopyToolGTK3(ToolBase):
    keymap = 'ctrl+c'
    name = 'Copy'
    description = 'Copy canvas'
    # It is not added to the toolbar as a button
    position = None

    def trigger(self, event):
        from gi.repository import Gtk, Gdk
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        window = self.figure.canvas.get_window()
        x, y, width, height = window.get_geometry()
        pb = Gdk.pixbuf_get_from_window(window, x, y, width, height)
        clipboard.set_image(pb)


fig = plt.figure()
plt.plot([1, 2, 3])

# Add the custom tools that we created
fig.canvas.manager.navigation.add_tool(ListTools)
if matplotlib.rcParams['backend'] == 'GTK3Cairo':
    fig.canvas.manager.navigation.add_tool(CopyToolGTK3)

# Just for fun, lets remove the back button
fig.canvas.manager.navigation.remove_tool('Back')

plt.show()
