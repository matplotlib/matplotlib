import matplotlib
matplotlib.use('GTK3Cairo')
# matplotlib.use('TkAGG')
matplotlib.rcParams['toolbar'] = 'navigation'
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase
from pydispatch import dispatcher

# Create a simple tool to list all the tools
class ListTools(ToolBase):
    # keyboard shortcut
    keymap = 'm'
    description = 'List Tools'
 
    def trigger(self, event):
        tools = self.navigation.get_tools()
  
        print ('_' * 80)
        print ("{0:12} {1:45} {2}".format('Name (id)',
                                          'Tool description',
                                          'Keymap'))
        print ('_' * 80)
        for name in sorted(tools.keys()):
            keys = ', '.join(sorted(tools[name]['keymap']))
            print ("{0:12} {1:45} {2}".format(name,
                                              tools[name]['description'],
                                              keys))
        print ('_' * 80)  
 
 
# A simple example of copy canvas
# ref: at https://github.com/matplotlib/matplotlib/issues/1987
class CopyToolGTK3(ToolBase):
    keymap = 'ctrl+c'
    description = 'Copy canvas'
    # It is not added to the toolbar as a button
    intoolbar = False
 
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
fig.canvas.manager.navigation.add_tool('List', ListTools)
if matplotlib.rcParams['backend'] == 'GTK3Cairo':
    fig.canvas.manager.navigation.add_tool('copy', CopyToolGTK3)
 
# # Just for fun, lets remove the forward button
# fig.canvas.manager.navigation.remove_tool('forward')


plt.show()
