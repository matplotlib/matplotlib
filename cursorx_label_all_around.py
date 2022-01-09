"""
    D:\OneDrive\myPython\mpl_interactive\mpl_cursor_cross_line_demo_2_SnaptoCursor_GD_x_limit_boundary_clipping
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import BoxStyle
from matplotlib.path import Path

'''
# https://stackoverflow.com/questions/36620025/pass-array-as-argument-in-python
def function(*args):
    print('type(args):{0}, args:{1}'.format(type(args), args))
    for u in args:
        print(u)

#Create a list of 5 elements
a = list(range(5))
print(a)

function(*a)
'''
class MyStyle3:
    """A simple box."""

    def __init__(self, pad=0.3):
        """
        The arguments must be floats and have default values.

        Parameters
        ----------
        pad : float
            amount of padding
        """
        self.pad = pad
        super().__init__()

    def __call__(self, x0, y0, width, height, mutation_size):
        """
        Given the location and size of the box, return the path of the box
        around it.

        Rotation is automatically taken care of.

        Parameters
        ----------
        x0, y0, width, height : float
            Box location and size.
        mutation_size : float
            Reference scale for the mutation, typically the text font size.
        """
        # padding
        pad = mutation_size * self.pad
        # width and height with padding added
        width = width
        height = height + 2.*pad
        # boundary of the padded box
        x0, y0 = x0 - pad, y0 - pad
        x1, y1 = x0 + width, y0 + height
        # print(f"{mutation_size} {pad}  tip point: {x1+pad}, {(y0+y1)/2.}")
        # return the new path
        # (x0, y0) : lower left corner of rectangle
        # (x1, y1) : upper right corner of rectangle
        return Path([(x0, y0),
                     (x1, y0), (x1,(y0+y1)*.25), (x1+pad, (y0+y1)/2.), (x1,(y0+y1)*.75),(x1, y1), 
                      (x0, y1), (x0, y0)], closed=True)


class MyStyle2:
    """A simple box."""

    def __init__(self, pad=0.3):
        """
        The arguments must be floats and have default values.

        Parameters
        ----------
        pad : float
            amount of padding
        """
        self.pad = pad
        super().__init__()

    def __call__(self, x0, y0, width, height, mutation_size):
        """
        Given the location and size of the box, return the path of the box
        around it.

        Rotation is automatically taken care of.

        Parameters
        ----------
        x0, y0, width, height : float
            Box location and size.
        mutation_size : float
            Reference scale for the mutation, typically the text font size.
        """
        # padding
        pad = mutation_size * self.pad
        # width and height with padding added
        width = width
        height = height + 2.*pad
        # boundary of the padded box
        x0, y0 = x0 + pad, y0 - pad
        x1, y1 = x0 + width, y0 + height
        # print(f"{mutation_size} {pad} tip point: {x0-pad}, {(y0+y1)/2.}")
        # return the new path
        # (x0, y0) : lower left corner of rectangle
        # (x1, y1) : upper right corner of rectangle
        return Path([(x0, y0),
                     (x1, y0), (x1, y1), (x0, y1),(x0,(y0+y1)*.75),
                     (x0-pad, (y0+y1)/2.), (x0,(y0+y1)*.25), (x0, y0)], closed=True)


class SnaptoCursor:
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """
    # def __init__(self, ax, fp, x, y):
    def __init__(self, ax, x, y):
    
        self.ax = ax
        # self.fp = fp
        self.lx = ax.axhline(color='k', lw=0.8, ls="--")  # the horiz line
        self.ly = ax.axvline(color='k', lw=0.8, ls="--")  # the vert line
        self.x = x
        self.y = y
        # print('type(x):{0}, x:{1}'.format(type(x), x[:3]))
        self.xm = len(x)                # KL
        # print(f'self.xm {self.xm}')
        # text location in axes coords
        self.dispTransmission = False
        # self.txt = ax.text( 0.5, 0.5, '',size=24, fontdict=None, transform=ax.transAxes)
        # self.txt = ax.text( 0.5, 0.5, '',size=24, fontdict={'fontproperties':self.fp }, transform=ax.transAxes)
        self.txt = ax.text( 0.5, 0.5, '',size=12, transform=ax.transAxes)

        self.crossMarker, = ax.plot(0, 0, linewidth=0, marker="o", color="red", markersize=np.sqrt(300), markerfacecolor='none', markeredgewidth=3)

        # along y-axis
        self.tvL = ax.text(0.5, 0.8, "", size=15, va="center", ha="left", rotation=0, bbox=dict(boxstyle="rpointy, pad=0.3", fc="yellow", alpha=0.7))
        self.tvR = ax.text(0.5, 0.8, "", size=15, va="center", ha="left", rotation=0, bbox=dict(boxstyle="lpointy, pad=0.3", fc="green", alpha=0.7))
        self.tvCx = ax.text(0.5, 0.8, "", size=15, va="center", ha="left", color='w', rotation=0, bbox=dict(boxstyle="lpointy, pad=0.3", fc="m", alpha=0.7))
        
        # along x-axis
        self.th = ax.text(
            0, 0, '', ha="center", va="top", rotation=0, size=15, color='w',
            bbox=dict(boxstyle="rarrow,pad=0.0", fc="none", ec="b", lw=2))
        bb = self.th.get_bbox_patch()
        bb.set_boxstyle("square", pad=0.0)
        self.th.set_bbox(dict(alpha=0.5, fc="red", ec="none", lw=2))


    def on_mouse_move(self, event):

        if not event.inaxes: return

        x, y = event.xdata, event.ydata

        indx = np.searchsorted(self.x, [x])[0]
        # indx = min(np.searchsorted(self.x, [x])[0], len(self.x) - 1)    # prevents an out-of-range error at the last point.
        # OR KL's approach
        if indx < self.xm:
            x = self.x[indx]
            y = self.y[indx]
        elif indx == self.xm:
            x = self.x[indx-1]
            y = self.y[indx-1]        
        
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        # print(x, y)
        self.crossMarker.set_data(x, y)

        fig.canvas.draw()
        bottom, top = ax.get_ylim()
        dy = (top-bottom)*0.01
        self.th.set_text("{0:.3f}".format(x))
        self.th.set_position((x, bottom-dy))

        left, right = ax.get_xlim()
        dx = (right-left)*0.01
        
        self.tvL.set_text("{0:.3f} ".format(y))    # right-pointy
        self.tvL.set_position((left, y))           # left of y-axis
        self.tvL.set_ha('right')                   # right text alignment

        self.tvR.set_text(" {0:.3f}".format(y))  # left-pointy
        self.tvR.set_position((right, y))         # Rightside of y-axis
        self.tvR.set_ha('left')                    # left text alignment
        
        self.tvCx.set_text(" {0:.3f}\n {1:.3f}".format(x, y))
        self.tvCx.set_position((x+3*dx, y))         # cross point x-offset +3*dx (marker size)
        self.tvCx.set_ha('left')                    # left text alignment
        
        if not self.dispTransmission:
            self.txt.set_text('x={0:,.3f}, y={1:,.3f}'.format(x, y))
            # print ('x=%1.2f, y=%1.2f'%(x,y))
            # self.ax.figure.canvas.draw()
            # self.txt.set_position((x,y))
            self.txt.set_position((0.5, top-dy))
        else:
            self.txt.set_text( 'y = {0:,.4f} @ {1:,.2f} nm'.format(y*100, x))
            # print ('x=%1.2f, y=%1.2f'%(x,y))
            # self.ax.figure.canvas.draw()
            # self.txt.set_position((x,y))
            self.txt.set_position((0.5, 0.5))
        self.ax.figure.canvas.draw_idle()

if __name__ == "__main__": 

    BoxStyle._style_list["lpointy"] = MyStyle2  # Register the custom style.
    BoxStyle._style_list["rpointy"] = MyStyle3  # Register the custom style.

    t = np.arange(0.0, 1.0, 0.01)
    s = np.sin(2*2*np.pi*t)
    fig, ax = plt.subplots()

    # https://matplotlib.org/3.2.2/gallery/userdemo/custom_boxstyle02.html

    '''
    https://scriptverse.academy/tutorials/python-matplotlib-plot-sine.html
    x = np.linspace(-np.pi,np.pi,100)

    p = np.sin(x) # y = sin(x)
    q = np.sin(2*x) # y = sin(2x)
    r = np.sin(3*x) # y = sin(3x)

    ## OR
    # 100 linearly spaced numbers
    x = np.linspace(-20,20,500)

    y = np.sin(x)/x # y = sin(x)/x
    y2 = 2*y
    y3 = -y
    '''
    # cursor = Cursor(ax)
    cursor = SnaptoCursor(ax, t, s)
    fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
    # fig.canvas.mpl_connect('button_press_event', cursor.onclick)
    # OR plt.connect('motion_notify_event', self)
    # OR fig.canvas.mpl_connect("motion_notify_event", on_focus) 
    # fig.canvas.mpl_connect('motion_notify_event', fig.canvas.onHilite)
    ax.plot(t, s, 'o')
    # plt.axis([0,1,-1,1])
    plt.show()