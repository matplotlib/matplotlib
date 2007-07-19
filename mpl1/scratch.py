
class AggPath(Path):

    agg_fillcolor = mtraits.ColorAgg(RendererAgg.blue)
    agg_strokecolor = mtraits.ColorAgg(RendererAgg.black)
    agg_path = mtraits.PathAgg 


    def __init__(self, path):
        self.strokecolor = path.strokecolor
        self.fillcolor = path.fillcolor
        self.alpha = path.alpha
        self.linewidth = path.linewidth
        self.antialiased = path.antialiased
        self.pathdata = path.pathdata
        self.affine = path.affine

        
        path.sync_trait('strokecolor', self)
        path.sync_trait('fillcolor', self)
        path.sync_trait('alpha', self)
        path.sync_trait('linewidth', self)
        path.sync_trait('antialiased', self)
        path.sync_trait('pathdata', self)
        path.sync_trait('affine', self)

    def _pathdata_changed(self, olddata, newdata):
        print 'pathdata changed'
        MOVETO, LINETO, CLOSEPOLY = Path.MOVETO, Path.LINETO, Path.CLOSEPOLY
        agg_path = agg.path_storage()
        codes, verts = newdata
        N = len(codes)
        for i in range(N):
            x, y = verts[i]
            code = codes[i]
            if code==MOVETO:
                agg_path.move_to(x, y)
            elif code==LINETO:
                agg_path.line_to(x, y)                
            elif code==CLOSEPOLY:
                agg_path.close_polygon()
        
        self.agg_path = agg_path
        
    def _fillcolor_changed(self, oldcolor, newcolor):        
        self.agg_fillcolor = self.color_to_rgba8(newcolor)

    def _strokecolor_changed(self, oldcolor, newcolor):                

        c = self.color_to_rgba8(newcolor)
        print 'stroke change: old=%s, new=%s, agg=%s, ret=%s'%(
            oldcolor, newcolor, self.agg_strokecolor, c)
        self.agg_strokecolor = c


    def color_to_rgba8(self, color):
        if color is None: return None
        rgba = [int(255*c) for c in color.r, color.g, color.b, color.a]
        return agg.rgba8(*rgba)


class Axis(Artist):
    tickcolor = mtraits.color('black')
    axiscolor = mtraits.color('black')
    tickwidth = mtraits.linewidth(0.5)
    viewlim   = mtraits.interval
    tickpath  = mtraits.path
    axispath  = mtraits.path
    
    def __init__(self, figure):        
        self.figure = figure
        self.pathids = set()
        
class XAxis(Axis):
    def __init__(self, figure, **kwargs):
        Axis.__init__(self, figure, **kwargs)

    def set_ticks(self, yloc, ysize, ticks, fmt):
        # we'll deal with locators, formatter and autoscaling later...
        # todo, remove old paths

        for pathid in self.pathids:
            self.figure.remove_path(pathid)
            
        codes = []
        verts = []
        tickmin = yloc-ysize/2.
        tickmax = yloc+ysize/2.
        for tick in ticks:
            codes.append(Path.MOVETO)
            verts.append((tick, tickmin))
            codes.append(Path.LINETO)
            verts.append((tick, tickmax))
        

        path = Path()
        path.verts = npy.array(verts)
        path.codes = npy.array(codes)
        path.strokecolor = self.tickcolor
        path.fillcolor = None
        path.linewidth = self.tickwidth
        path.antialiased = False

        self.pathids.add(self.figure.add_path(path))

        
        xmin, xmax = self.viewlim

        # the axis line
        codes = []
        verts = []
        codes.append(Path.MOVETO)
        verts.append((xmin, yloc))
        codes.append(Path.LINETO)
        verts.append((xmax, yloc))

        path = Path()
        path.verts = npy.array(verts)
        path.codes = npy.array(codes)
        path.strokecolor = self.axiscolor
        path.fillcolor = None
        path.antialiased = False
        
        self.pathids.add(self.figure.add_path(path))

    
class YAxis:
    def __init__(self, figure):
        Axis.__init__(self, figure)

    def set_ticks(self, xloc, xsize, ticks, fmt):

        for pathid in self.pathids:
            self.figure.remove_path(pathid)

        codes = []
        verts = []
        tickmin = yloc-ysize/2.
        tickmax = yloc+ysize/2.
        for tick in ticks:
            codes.append(Path.MOVETO)
            verts.append((tickmin, tick))
            codes.append(Path.LINETO)
            verts.append((tickmax, tick))
        

        self.tickpath = path = Path()
        path.verts = npy.array(verts)
        path.codes = npy.array(codes)
        path.strokecolor = self.tickcolor
        path.fillcolor = None
        path.linewidth = self.tickwidth
        path.antialiased = False

        self.pathids.add(self.figure.add_path(path))

        
        ymin, ymax = self.viewlim

        # the axis line
        codes = []
        verts = []
        codes.append(Path.MOVETO)
        verts.append((xloc, ymin))
        codes.append(Path.LINETO)
        verts.append((xloc, ymax))

        self.axispath = path = Path()
        path.verts = npy.array(verts)
        path.codes = npy.array(codes)
        path.strokecolor = self.axiscolor
        path.fillcolor = None
        path.antialiased = False
        
        self.pathids.add(self.figure.add_path(path))






if 0:
    ax1.set_ylim(-1.1, 1.1)

    xaxis =  XAxis(ax1)
    xaxis.set_ticks(0, 0.1, npy.arange(11.0), '%d')

    yaxis1 = YAxis(ax1)
    yaxis1.set_ticks(-1.1, 0.2, npy.arange(-1.0, 1.1, 0.5), '%d')
    yaxis1.axiscolor = line1.color

    yaxis2 = YAxis(ax1)
    yaxis2.set_ticks(5.0, 0.2, npy.arange(-1.0, 1.1, 0.5), '%d')





    theta = 0.25*npy.pi  # 45 degree axes rotation
    #rotate_axes(ax1, theta)


    r = npy.arange(0, 1, 0.01)
    theta = r*4*npy.pi
    X2 = npy.array([r,theta]).T
    line2 = Line(X2, model=Polar())
    ax2.add_line(line2)
    # currently cartesian
    ax2.set_xlim(-1,1)
    ax2.set_ylim(-1,1)
