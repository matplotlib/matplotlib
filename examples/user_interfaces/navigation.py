import matplotlib
matplotlib.use('GTK3Cairo')
#matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3], label='My First line')
ax.plot([2, 3, 4], label='Second line')



from matplotlib.backend_bases import ToolBase
class ListTools(ToolBase):
    #keyboard shortcut
    keymap = 'm'
    #Name used as id, must be unique between tools of the same navigation
    name = 'List' 
    description = 'List Tools' 
    #Where to put it in the toolbar, -1 = at the end, None = Not in toolbar
    position = -1
 
    def activate(self, event):
        #The most important attributes are navigation and figure
        self.navigation.list_tools()

#Add the simple tool to the toolbar
fig.canvas.manager.navigation.add_tool(ListTools)

#Just for fun, lets remove the back button    
fig.canvas.manager.navigation.remove_tool('Back')

#looking at https://github.com/matplotlib/matplotlib/issues/1987
#a simple example of copy canvas
class CopyTool(ToolBase):
    keymap = 'ctrl+c'
    name = 'Copy'
    description = 'Copy canvas'
    position = -1

    def activate(self, event):
        from gi.repository import Gtk, Gdk, GdkPixbuf
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        window = self.figure.canvas.get_window()
        x, y, width, height = window.get_geometry()
        pb = Gdk.pixbuf_get_from_window(window, x, y, width, height)
        clipboard.set_image(pb)

fig.canvas.manager.navigation.add_tool(CopyTool)

plt.show()