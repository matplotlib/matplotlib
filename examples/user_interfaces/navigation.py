'''This example demonstrates how the `matplotlib.backend_bases.NavigationBase`
class allows to:
* Modify the Toolbar
* Add tools
* Remove tools
'''


from __future__ import print_function
import matplotlib
matplotlib.use('GTK3Cairo')
matplotlib.rcParams['toolbar'] = 'navigation'
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase


class ListTools(ToolBase):
    '''List all the tools controlled by `Navigation`'''
    # keyboard shortcut
    keymap = 'm'
    description = 'List Tools'

    def trigger(self, *args, **kwargs):
        print('_' * 80)
        print("{0:12} {1:45} {2}".format('Name (id)',
                                         'Tool description',
                                         'Keymap'))
        print('-' * 80)
        tools = self.navigation.tools
        for name in sorted(tools.keys()):
            if not tools[name].description:
                continue
            keys = ', '.join(sorted(self.navigation.get_tool_keymap(name)))
            print("{0:12} {1:45} {2}".format(name,
                                             tools[name].description,
                                             keys))
        print('_' * 80)
        print("Active Toggle tools")
        print("{0:12} {1:45}").format("Group", "Active")
        print('-' * 80)
        for group, active in self.navigation.active_toggle.items():
            print("{0:12} {1:45}").format(group, active)


# ref: at https://github.com/matplotlib/matplotlib/issues/1987
class CopyToolGTK3(ToolBase):
    '''Copy canvas to clipboard'''
    keymap = 'ctrl+c'
    description = 'Copy canvas'

    def trigger(self, *args, **kwargs):
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
fig.canvas.manager.navigation.add_tool('copy', CopyToolGTK3)

# Add an existing tool to new group `foo`.
# It can be added as many times as we want
fig.canvas.manager.toolbar.add_tool('zoom', 'foo')

# Remove the forward button
fig.canvas.manager.navigation.remove_tool('forward')

# To add a custom tool to the toolbar at specific location
fig.canvas.manager.toolbar.add_tool('List', 'navigation', 1)

plt.show()
