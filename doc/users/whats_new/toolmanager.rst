ToolManager
-----------

Federico Ariza wrote the new `matplotlib.backend_managers.ToolManager` that comes as replacement for `NavigationToolbar2`

`ToolManager` offers a new way of looking at the user interactions with the figures.
Before we had the `NavigationToolbar2` with its own tools like `zoom/pan/home/save/...` and also we had the shortcuts like
`yscale/grid/quit/....`
`Toolmanager` relocate all those actions as `Tools` (located in `matplotlib.backend_tools`), and defines a way to `access/trigger/reconfigure` them.

The `Toolbars` are replaced for `ToolContainers` that are just GUI interfaces to `trigger` the tools. But don't worry the default backends include a `ToolContainer` called `toolbar`


.. note::
	For the moment the `ToolManager` is working only with `GTK3` and `Tk` backends.
	Make sure you are using one of those.
	Port for the rest of the backends is comming soon.
	
	To activate the `ToolManager` include the following at the top of your file:
	
	 >>> matplotlib.rcParams['toolbar'] = 'toolmanager'
	

Interact with the ToolContainer
```````````````````````````````

The most important feature is the ability to easily reconfigure the ToolContainer (aka toolbar).
For example, if we want to remove the "forward" button we would just do.

 >>> fig.canvas.manager.toolmanager.remove_tool('forward')

Now if you want to programmatically trigger the "home" button

 >>> fig.canvas.manager.toolmanager.trigger_tool('home')


New Tools
`````````

It is possible to add new tools to the ToolManager

A very simple tool that prints "You're awesome" would be::

    from matplotlib.backend_tools import ToolBase
    class AwesomeTool(ToolBase):
        def trigger(self, *args, **kwargs):
            print("You're awesome")


To add this tool to `ToolManager`

 >>> fig.canvas.manager.toolmanager.add_tool('Awesome', AwesomeTool)

If we want to add a shortcut ("d") for the tool

 >>> fig.canvas.manager.toolmanager.update_keymap('Awesome', 'd')


To add it to the toolbar inside the group 'foo'

 >>> fig.canvas.manager.toolbar.add_tool('Awesome', 'foo')


There is a second class of tools, "Toggleable Tools", this are almost the same as our basic tools, just that belong to a group, and are mutually exclusive inside that group.
For tools derived from `ToolToggleBase` there are two basic methods `enable` and `disable` that are called automatically whenever it is toggled.


A full example is located in :ref:`user_interfaces-toolmanager`
