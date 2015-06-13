"""
============
Tool Manager
============

This example demonstrates how to:

* Modify the Toolbar
* Create tools
* Add tools
* Remove tools

Using `matplotlib.backend_managers.ToolManager`
"""


from __future__ import print_function
import matplotlib
matplotlib.use('GTK3Cairo')
matplotlib.rcParams['toolbar'] = 'toolmanager'
import matplotlib.pyplot as plt
from matplotlib.backend_tools import (ToolBase, ToolToggleBase,
                                      add_tools_to_container)
from gi.repository import Gtk, Gdk
from random import uniform


class ListTools(ToolBase):
    '''List all the tools controlled by the `ToolManager`'''
    # keyboard shortcut
    default_keymap = 'm'
    description = 'List Tools'

    def trigger(self, *args, **kwargs):
        print('_' * 80)
        print("{0:12} {1:45} {2}".format(
            'Name (id)', 'Tool description', 'Keymap'))
        print('-' * 80)
        tools = self.toolmanager.tools
        for name in sorted(tools):
            if not tools[name].description:
                continue
            keys = ', '.join(sorted(self.toolmanager.get_tool_keymap(name)))
            print("{0:12} {1:45} {2}".format(
                name, tools[name].description, keys))
        print('_' * 80)
        print("Active Toggle tools")
        print("{0:12} {1:45}".format("Group", "Active"))
        print('-' * 80)
        for group, active in self.toolmanager.active_toggle.items():
            print("{0:12} {1:45}".format(str(group), str(active)))


class GroupHideTool(ToolToggleBase):
    '''Show lines with a given gid'''
    default_keymap = 'G'
    description = 'Show by gid'
    default_toggled = True

    def __init__(self, *args, **kwargs):
        self.gid = kwargs.pop('gid')
        ToolToggleBase.__init__(self, *args, **kwargs)

    def enable(self, *args):
        self.set_lines_visibility(True)

    def disable(self, *args):
        self.set_lines_visibility(False)

    def set_lines_visibility(self, state):
        for ax in self.figure.get_axes():
            for line in ax.get_lines():
                if line.get_gid() == self.gid:
                    line.set_visible(state)
        self.figure.canvas.draw()


class LineTool(ToolBase):
    description = 'Draw a random line'

    def __init__(self, *args, **kwargs):
        self.color = kwargs.pop('color')
        ToolBase.__init__(self, *args, **kwargs)

    def trigger(self, *args, **kwargs):
        x0, y0, x1, y1 = (uniform(0, 2), uniform(1, 4), uniform(0, 2),
                         uniform(1, 4))
        fig = self.figure
        fig.gca().plot([x0, x1], [y0, y1], color=self.color, gid=self.color)
        fig.canvas.draw_idle()


class DotTool(ToolBase):
    description = 'Draw a random dot'

    def __init__(self, *args, **kwargs):
        self.color = kwargs.pop('color')
        ToolBase.__init__(self, *args, **kwargs)

    def trigger(self, *args, **kwargs):
        x0, y0 = uniform(0, 2), uniform(1, 4)
        fig = self.figure
        fig.gca().plot([x0], [y0], 'o', color=self.color, gid=self.color)
        fig.canvas.draw_idle()


fig = plt.figure()
plt.plot([1, 2, 3], gid='mygroup')
plt.plot([2, 3, 4], gid='unknown')
plt.plot([3, 2, 1], gid='mygroup')

# Add the custom tools that we created
manager = fig.canvas.manager
tool_mgr = manager.toolmanager
tool_mgr.add_tool('List', ListTools)
tool_mgr.add_tool('Hide', GroupHideTool, gid='mygroup')


# Add an existing tool to new group `foo`.
# It can be added as many times as we want
manager.toolbar.add_tool('zoom', 'foo')

# Remove the forward button
tool_mgr.remove_tool('forward')

# To add a custom tool to the toolbar at specific location inside
# the navigation group
manager.toolbar.add_tool('Hide', 'navigation', 1)

for i, c in enumerate(['yellowgreen', 'forestgreen']):
    sidebar = manager.backend.Toolbar(tool_mgr)
    sidebar.set_flow('vertical')
    tools = [['shapes', [tool_mgr.add_tool('L%s' % i, LineTool, color=c),
                         tool_mgr.add_tool('D%s' % i, DotTool, color=c)]],
             ['hide', [tool_mgr.add_tool('H%s' % i, GroupHideTool, gid=c)]]]

    manager.window.add_element(sidebar, False, 'west')
    add_tools_to_container(sidebar, tools)

plt.show()
