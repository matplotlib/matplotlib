"""
This is an example to show how to build cross-GUI applications using
matplotlib event handling to interact with objects on the canvas

"""
from matplotlib.patches import Polygon
from matplotlib.numerix import sqrt, nonzero, equal, asarray, dot, Float
from matplotlib.numerix.mlab import amin
from matplotlib.mlab import dist_point_to_segment



        
class EditablePolygon(Polygon):
    """
    An editable polygon.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point      

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices
          
    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit
    def __init__(self, *args, **kwargs):
        Polygon.__init__(self, *args, **kwargs)        
        self.line = Line2D([],[],marker='o', markerfacecolor='r')
        self._ind = None # the active vert

        self.xy = list(self.xy)  # make sure it is editable
        
    def set_figure(self, fig):
        Polygon.set_figure(self, fig)
        self.line.set_figure(fig)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.figure.canvas.mpl_connect('key_press_event', self.key_press_callback)        
        self.figure.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.figure.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)                
        
        
    def set_transform(self, trans):
        Polygon.set_transform(self, trans)
        self.line.set_transform(trans)

    def set_clip_on(self, b):
        Polygon.set_clip_on(self, b)
        self.line.set_clip_on(b)

    def set_clip_box(self, b):
        Polygon.set_clip_box(self, b)
        self.line.set_clip_box(b)


    def draw(self, renderer):
        if not self._visible: return 
        Polygon.draw(self, renderer)
        if self.showverts:            
            self.line.set_data(zip(*self.xy))
            self.line.draw(renderer)

    def get_ind_under_point(self, event):
        x, y = zip(*self.xy)
        # display coords        
        xt, yt = self._transform.numerix_x_y(x, y)
        d = sqrt((xt-event.x)**2 + (yt-event.y)**2)
        indseq = nonzero(equal(d, amin(d)))
        ind = indseq[0]
        if d[ind]>=self.epsilon:
            ind = None

        return ind
        
    def button_press_callback(self, event):
        if not self.showverts: return 
        if event.inaxes==None: return
        self._ind = self.get_ind_under_point(event)
        

    def button_release_callback(self, event):
        if not self.showverts: return 
        self._ind = None

    def key_press_callback(self, event):
        if not event.inaxes: return
        if event.key=='t':
            self.showverts = not self.showverts
            if not self.showverts: self._ind = None
        elif event.key=='d':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.xy = [tup for i,tup in enumerate(self.xy) if i!=ind]
        elif event.key=='i':            
            xys = self._transform.seq_xy_tups(self.xy)
            p = event.x, event.y # display coords
            for i in range(len(xys)-1):                
                s0 = xys[i]
                s1 = xys[i+1]
                d = dist_point_to_segment(p, s0, s1)
                if d<=self.epsilon:
                    self.xy.insert(i+1, (event.xdata, event.ydata))
                    break
                
                


        self.figure.canvas.draw()

    def motion_notify_callback(self, event):
        if not self.showverts: return 
        if self._ind is None: return
        if event.inaxes is None: return
        x,y = event.xdata, event.ydata
        self.xy[self._ind] = x,y
        self.figure.canvas.draw_idle()


from pylab import *
verts = Circle((.5,.5),.5).get_verts()
p = EditablePolygon(verts)

fig = figure()
ax = subplot(111)
ax.add_patch(p)
title('Click and drag a point to move it')
axis([0,1,0,1])
show()
