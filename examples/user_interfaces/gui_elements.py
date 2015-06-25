'''This example demonstrates how to:
* Create new toolbars
* Create new windows
Using `matplotlib.backend_managers.ToolManager`,
`matplotlib.backend_bases.WindowBase` and
`matplotlib.backend_bases.ToolContainerBase`
'''

from __future__ import print_function
import matplotlib
matplotlib.use('GTK3Cairo')
matplotlib.rcParams['toolbar'] = 'toolmanager'
import matplotlib.pyplot as plt

fig = plt.figure()

# Shortcuts to FigureManager and ToolManager
manager = fig.canvas.manager
tool_mgr = manager.toolmanager

# Create a new toolbar
topbar = manager.backend.Toolbar(tool_mgr)
# The options are north, east, west and south
manager.window.add_element(topbar, False, 'north')

# Remove some tools from the main toolbar and add them to the
# new sidebar
for tool in ('home', 'back', 'forward'):
    manager.toolbar.remove_toolitem(tool)
    topbar.add_tool(tool, None)

plt.plot([1, 2, 3])

# Add a new window
win = manager.backend.Window('Extra tools')
# create a sidebar for the new window
sidebar = manager.backend.Toolbar(tool_mgr)
# set the direction of the sidebar
# the options are horizontal and vertical
sidebar.set_flow('vertical')
# add the sidebar to the new window
win.add_element(sidebar, False, 'west')

# Add some tools to the new sidebar
for tool in ('home', 'back', 'forward', 'zoom', 'pan'):
    sidebar.add_tool(tool, None)
# show the new window
win.show()

plt.show()
