Using Matplotlib with QtDesigner
--------------------------------
QtDesigner is a "WYSIWYG" editor for creating Qt user interfaces. In addition
to the widgets packaged in the Qt library, there is support for custom PyQt5
widgets to be made available within this framework as well. The addition of the
``FigureDesignerPlugin`` makes it possible to include a ``FigureCanvasQt``
widget  by simply dragging and dropping in the QtDesigner interface. The
generated XML file can then be loaded using ``pyuic`` and populated with data
using standard ``matplotlib`` syntax.

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~
Before using the ``FigureDesignerPlugin`` you need to make sure you are in a
compatible environment. First off, ``PyQt5`` has the option to be installed
without QtDesigner. Make sure the installation you were using has not excluded
the Designer. 

We will also need to set the ``$PYQTDESIGNERPATH`` environment variable to
properly locate our custom widget plugin. This informs the QtDesigner that we
want the ``FigureCanvas`` widget as an option while we are creating our screen.
On Linux this would look like the example below with your own path to the
``matplotlib`` source code substituted in place.

.. code:: bash

   export PYQTDESIGNERPATH=$PYQTDESIGNERPATH:/path/to/matplotlib/lib/matplotlib/mpl-data

For more information consult the `official PyQt
<http://pyqt.sourceforge.net/Docs/PyQt5/designer.html#writing-qt-designer-plugins>`_
documentation. If you are unsure where to find the ``mpl-data`` folder you can
refer to the ``matplotlib.rcParams['datapath']``

Usage
~~~~~
The general process for using the ``QtDesigner`` and ``matplotlib`` is
explained below:

1. If your environment is configured correctly you should see the
   ``FigureCanvasQt`` widget in the left hand column of your QtDesigner
   interface.  It can now be used as if it were any other widget, place it in
   its desired location and give it a meaningful ``objectName`` for reference
   later. The code below assumes you called it "example_plot"

2. Once you are done creating your interface in Designer it is time to load our
   the ``.ui`` file created and manipulate the ``matplotlib.Figure``. The
   simplest way is to use the ``uic`` to load the ``.ui`` file into a custom
   widget. This will make our ``FigureCanvasQt`` object we created in Designer
   available to us.

   .. code:: python

        from PyQt5 import uic
        from PyQt5.QtWidgets import QWidget

        # Create a QWidget to contain our created display
        my_widget = QWidget()
        # Load the UI we created in Designer
        uic.loadUi('path/to/my_file.ui', widget)
        # We now access to the Figure we created in Designer
        my_widget.example_plot.figure

3. Now use standard ``matplotlib`` syntax to add axes and data to the
   ``Figure``.

   .. code:: python

      ax = my_widget.example_plot.figure.add_subplot(1,1,1)
      ax.plot([1, 2, 3], [3, 2, 1])
