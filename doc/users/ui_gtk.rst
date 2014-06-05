.. _ui_Gtk:

Gtk+
====
Gtk+ or Gnome-toolkit is a graphical user interfaces toolkit with different language bindings. The resulting user interface is designed directly within the program code (as opposed to using a graphical-user-interface-designer like Glade). This works well for small applications with simple user interfaces. 

.. _ui_gtk3_python3:

Gtk+ 3 example using Python 3
---------------------------------
This example uses the Python-3-bindings of Gtk. In order to understand the code it is recomended to work through the Python-GtK3-Tutorial: http://python-gtk-3-tutorial.readthedocs.org/en/latest/.

The code for a simple window with plot looks like this:

::

        #!/usr/bin/env python
        """
        demonstrate adding a FigureCanvasGTK3Agg widget to a Gtk.ScrolledWindow
        using GTK3 accessed via pygobject
        """

        from gi.repository import Gtk

        from matplotlib.figure import Figure
        from numpy import arange, sin, pi
        from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

        win = Gtk.Window()
        win.connect("delete-event", Gtk.main_quit )
        win.set_default_size(400,300)
        win.set_title("Embedding in GTK")

        f = Figure(figsize=(5,4), dpi=100)
        a = f.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)
        a.plot(t,s)

        sw = Gtk.ScrolledWindow()
        win.add (sw)
        # A scrolled window border goes outside the scrollbars and viewport
        sw.set_border_width (10)

        canvas = FigureCanvas(f)  # a Gtk.DrawingArea
        canvas.set_size_request(800,600)
        sw.add_with_viewport (canvas)

        win.show_all()
        Gtk.main()

The same example showing the Matplotlib-toolbar:

::

        #!/usr/bin/env python
        """
        demonstrate NavigationToolbar with GTK3 accessed via pygobject
        """

        from gi.repository import Gtk

        from matplotlib.figure import Figure
        from numpy import arange, sin, pi
        from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
        from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar

        win = Gtk.Window()
        win.connect("delete-event", Gtk.main_quit )
        win.set_default_size(400,300)
        win.set_title("Embedding in GTK")

        f = Figure(figsize=(5,4), dpi=100)
        a = f.add_subplot(1,1,1)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)
        a.plot(t,s)

        vbox = Gtk.VBox()
        win.add(vbox)

        # Add canvas to vbox
        canvas = FigureCanvas(f)  # a Gtk.DrawingArea
        vbox.pack_start(canvas, True, True, 0)

        # Create toolbar
        toolbar = NavigationToolbar(canvas, win)
        vbox.pack_start(toolbar, False, False, 0)

        win.show_all()
        Gtk.main()
 




