from matplotlib.matlab import *


class Cursor:
    def __init__(self, canvas, ax):
        self.canvas = canvas
        self.ax = ax
        self.lx, = ax.plot( (0,0), (0,0), 'k-' )  # the horiz line
        self.ly, = ax.plot( (0,0), (0,0), 'k-' )  # the vert line

        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)
        
    def mouse_move(self, widget, event):
        height = self.ax.figure.bbox.height()
        x, y = event.x, height-event.y

        if self.ax.in_axes(x, y):
            # transData transforms data coords to display coords.  Use
            # the inverse method to transform back to data coords then
            # update the line

            # the cursor position
            x, y =  ax.transData.inverse_xy_tup( (x,y) )            
            # the view limits
            minx, maxx = ax.viewLim.intervalx().get_bounds()
            miny, maxy = ax.viewLim.intervaly().get_bounds()

            # update the line positions
            self.lx.set_data( (minx, maxx), (y, y) )
            self.ly.set_data( (x, x), (miny, maxy) )

            self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
            self.canvas.draw()


class SnaptoCursor:
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """
    def __init__(self, canvas, ax, x, y):
        self.canvas = canvas
        self.ax = ax
        self.lx, = ax.plot( (0,0), (0,0), 'k-' )  # the horiz line
        self.ly, = ax.plot( (0,0), (0,0), 'k-' )  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        self.txt = ax.text( 0.7, 0.9, '', transform=ax.transAxes)
        
    def mouse_move(self, widget, event):
        height = self.ax.figure.bbox.height()
        x, y = event.x, height-event.y

        if self.ax.in_axes(x, y):
            # transData transforms data coords to display coords.  Use
            # the inverse method to transform back to data coords then
            # update the line

            # the cursor position
            x, y =  ax.transData.inverse_xy_tup( (x,y) )            
            # the view limits
            minx, maxx = ax.viewLim.intervalx().get_bounds()
            miny, maxy = ax.viewLim.intervaly().get_bounds()

            indx = searchsorted(self.x, [x])[0]
            x = self.x[indx]
            y = self.y[indx]            
            # update the line positions
            self.lx.set_data( (minx, maxx), (y, y) )
            self.ly.set_data( (x, x), (miny, maxy) )

            self.txt.set_text( 'x=%1.2f, y=%1.2f'%(x,y) )
            print 'x=%1.2f, y=%1.2f'%(x,y)
            self.canvas.draw()

t = arange(0.0, 1.0, 0.01)
s = sin(2*2*pi*t)
ax = subplot(111)

canvas = get_current_fig_manager().canvas
#cursor = Cursor(canvas, ax)
cursor = SnaptoCursor(canvas, ax, t, s) 
canvas.connect('motion_notify_event', cursor.mouse_move)

ax.plot(t, s, 'o')
axis([0,1,-1,1])
show()
