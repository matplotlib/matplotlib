"""
Show how to use the mouse to select objects and a build dialog to set
line properties.  The approach here can be readily extended to include
all artists and properties.  Volunteers welcome!
"""

from __future__ import division
from Numeric import sin, pi, arange, absolute, sqrt

import matplotlib
matplotlib.use('GTK')

from matplotlib.backends.backend_gtk import NavigationToolbar, \
     error_msg, colorManager, FigureCanvasGTK
from matplotlib.axes import Subplot

from matplotlib.figure import Figure
from matplotlib.lines import Line2D, lineStyles, lineMarkers
from matplotlib.transforms import Bound2D
from matplotlib.patches import draw_bbox
from matplotlib.colors import colorConverter
import gtk


def get_color(rgb):
    dialog = gtk.ColorSelectionDialog('Choose color')

    colorsel = dialog.colorsel
    color = colorManager.get_color(rgb)
    colorsel.set_previous_color(color)
    colorsel.set_current_color(color)
    colorsel.set_has_palette(gtk.TRUE)

    response = dialog.run()

    if response == gtk.RESPONSE_OK:
        rgb = colorManager.get_rgb(colorsel.get_current_color())
    else: 
        rgb = None
    dialog.destroy()
    return rgb



def make_option_menu( names, func=None ):
    """
    Make an option menu with list of names in names.  Return value is
    a optMenu, itemDict tuple, where optMenu is the option menu and
    itemDict is a dictionary mapping menu items to labels.  Eg

    optmenu, menud = make_option_menu( ('Bill', 'Ted', 'Fred') )

    ...set up dialog ...
    if response==gtk.RESPONSE_OK:
       item = optmenu.get_menu().get_active()
       print menud[item]  # this is the selected name


    if func is not None, call func with label when selected
    """
    optmenu = gtk.OptionMenu()
    optmenu.show()
    menu = gtk.Menu()
    menu.show()
    d = {}
    for label in names:
        item = gtk.MenuItem(label)
        menu.append(item)
        item.show()
        d[item] = label
        if func is not None:
            item.connect("activate", func, label)
    optmenu.set_menu(menu)
    return optmenu, d




class LineDialog(gtk.Dialog):
    def __init__(self, line, fig):
        gtk.Dialog.__init__(self, 'Line Properties')

        self.fig = fig
        self.line = line

        table = gtk.Table(3,2)
        table.show()
        table.set_row_spacings(4)
        table.set_col_spacings(4)
        table.set_homogeneous(True)
        self.vbox.pack_start(table, gtk.TRUE, gtk.TRUE)

        row = 0


        
        label = gtk.Label('linewidth')
        label.show()
        entry = gtk.Entry()
        entry.show()
        entry.set_text(str(line.get_linewidth()))
        self.entryLineWidth = entry
        table.attach(label, 0, 1, row, row+1,
                     xoptions=gtk.FALSE, yoptions=gtk.FALSE)
        table.attach(entry, 1, 2, row, row+1,
                     xoptions=gtk.TRUE, yoptions=gtk.FALSE)
        row += 1


        self.rgbLine = colorConverter.to_rgb(self.line.get_color())
        
        def set_color(button):
            rgb = get_color(self.rgbLine)
            if rgb is not None:
                self.rgbLine = rgb
                
        label = gtk.Label('color')
        label.show()
        button = gtk.Button(stock=gtk.STOCK_SELECT_COLOR)
        button.show()
        button.connect('clicked', set_color)
        table.attach(label, 0, 1, row, row+1,
                     xoptions=gtk.FALSE, yoptions=gtk.FALSE)
        table.attach(button, 1, 2, row, row+1,
                     xoptions=gtk.TRUE, yoptions=gtk.FALSE)
        row += 1

        
        ## line styles
        label = gtk.Label('linestyle')
        label.show()
        thisStyle = line.get_linestyle()
        styles = [thisStyle]
        for key in lineStyles.keys():
            if key == thisStyle: continue
            styles.append(key)

        self.menuLineStyle, self.menuLineStyleItemd = make_option_menu(styles)
        table.attach(label, 0, 1, row, row+1,
                     xoptions=gtk.FALSE, yoptions=gtk.FALSE)
        table.attach(self.menuLineStyle, 1, 2, row, row+1,
                     xoptions=gtk.TRUE, yoptions=gtk.FALSE)
        row += 1

        ## marker
        label = gtk.Label('marker')
        label.show()

        keys = lineMarkers.keys()
        keys.append('None')
        marker = line.get_marker()
        if marker is None: marker = 'None'        
        styles = [marker]
        for key in lineMarkers.keys():
            if key == marker: continue
            styles.append(key)

        self.menuMarker, self.menuMarkerItemd = make_option_menu(styles)
        table.attach(label, 0, 1, row, row+1,
                     xoptions=gtk.FALSE, yoptions=gtk.FALSE)
        table.attach(self.menuMarker, 1, 2, row, row+1,
                     xoptions=gtk.TRUE, yoptions=gtk.FALSE)
        row += 1
        
        
        
        self.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
        self.add_button(gtk.STOCK_APPLY, gtk.RESPONSE_APPLY)
        self.add_button(gtk.STOCK_OK, gtk.RESPONSE_OK)


    def update_line(self):
        s = self.entryLineWidth.get_text()
        try: lw = float(s)
        except ValueError:
            error_msg('Line width must be a float.  You entered %s' % s)
        else:
            self.line.set_linewidth(lw)


        item = self.menuLineStyle.get_menu().get_active()
        style = self.menuLineStyleItemd[item]
        self.line.set_linestyle(style)

        item = self.menuMarker.get_menu().get_active()
        style = self.menuMarkerItemd[item]
        if style is 'None': style = None
        self.line.set_marker(style)

        self.line.set_color(self.rgbLine)
        self.fig.draw()

    def run(self):
        while 1:
            response = gtk.Dialog.run(self)
            if response==gtk.RESPONSE_APPLY:
                self.update_line()
            elif response==gtk.RESPONSE_OK:
                self.update_line()
                break
            elif response==gtk.RESPONSE_CANCEL:
                break
        self.destroy()            

class PickerCanvas(FigureCanvasGTK):

    def button_press_event(self, widget, event):
        width, height = self.figure.renderer.gdkDrawable.get_size()
        
        self.pick(event.x, height-event.y)

    def select_line(self, line):        
        dlg = LineDialog(line, self)
        dlg.show()
        dlg.run()
        
    def select_text(self, text):
        print 'select text', text.get_text()



    def pick(self, x, y, epsilon=5):
        """
        Return the artist at location x,y with an error tolerance epsilon
        (in pixels)
        """

        clickBBox = Bound2D(x-epsilon/2, y-epsilon/2, epsilon, epsilon)
        draw_bbox(self.figure.dpi, clickBBox, self.figure.renderer)

        def over_text(t):
            bbox = t.get_window_extent(self.figure.renderer)
            return clickBBox.overlap(bbox)

        def over_line(line):
            # can't use the line bbox because it covers the entire extent
            # of the line
            xdata = line.transx.positions(line.get_xdata())
            ydata = line.transy.positions(line.get_ydata())
            distances = sqrt((x-xdata)**2 + (y-ydata)**2)
            return min(distances)<epsilon

        for ax in self.figure.axes:

            for line in ax.get_lines():
                if over_line(line):
                    self.select_line(line)
                    return

            text = ax.get_xticklabels()
            text.extend( ax.get_yticklabels() )
            
            for t in text:
                if over_text(t):
                    self.select_text(t)
                    return
                

                            
win = gtk.Window()
win.set_name("Embedding in GTK")
win.connect("destroy", gtk.mainquit)
win.set_border_width(5)

vbox = gtk.VBox(spacing=3)
win.add(vbox)
vbox.show()

fig = Figure(figsize=(5,4), dpi=100)

ax = Subplot(fig, 111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)

ax.plot(t,s)
ax.set_title('click on line or text')
fig.add_axis(ax)

canvas = PickerCanvas(fig)
canvas.show()
vbox.pack_start(canvas)

toolbar = NavigationToolbar(canvas, win)
toolbar.show()
vbox.pack_start(toolbar, gtk.FALSE, gtk.FALSE)


win.show()
gtk.mainloop()
