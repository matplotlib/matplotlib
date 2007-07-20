#!/usr/bin/python
# axis3d.py
#
# Created: 23 Sep 2005

import math

import lines
import axis
import patches
import text

import art3d
import proj3d

import numpy as npy

def norm_angle(a):
    """Return angle between -180 and +180"""
    a = (a+360)%360
    if a > 180: a = a-360
    return a

def text_update_coords(self, renderer):
    """Modified method update_coords from TextWithDash

    I could not understand the original text offset calculations and
    it gave bad results for the angles I was using.  This looks
    better, although the text bounding boxes look a little
    inconsistent
    """

    (x, y) = self.get_position()
    dashlength = self.get_dashlength()

    # Shortcircuit this process if we don't have a dash
    if dashlength == 0.0:
        self._mytext.set_position((x, y))
        return

    dashrotation = self.get_dashrotation()
    dashdirection = self.get_dashdirection()
    dashpad = self.get_dashpad()
    dashpush = self.get_dashpush()
    transform = self.get_transform()

    angle = text.get_rotation(dashrotation)

    theta = math.pi*(angle/180.0+dashdirection-1)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    # Compute the dash end points
    # The 'c' prefix is for canvas coordinates
    cxy = npy.array(transform.xy_tup((x, y)))
    cd = npy.array([cos_theta, sin_theta])
    c1 = cxy+dashpush*cd
    c2 = cxy+(dashpush+dashlength)*cd
    (x1, y1) = transform.inverse_xy_tup(tuple(c1))
    (x2, y2) = transform.inverse_xy_tup(tuple(c2))
    self.dashline.set_data((x1, x2), (y1, y2))

    # We now need to extend this vector out to
    # the center of the text area.
    # The basic problem here is that we're "rotating"
    # two separate objects but want it to appear as
    # if they're rotated together.
    # This is made non-trivial because of the
    # interaction between text rotation and alignment -
    # text alignment is based on the bbox after rotation.
    # We reset/force both alignments to 'center'
    # so we can do something relatively reasonable.
    # There's probably a better way to do this by
    # embedding all this in the object's transformations,
    # but I don't grok the transformation stuff
    # well enough yet.
    we = self._mytext.get_window_extent(renderer=renderer)
    w, h = we.width(), we.height()
    off = npy.array([cos_theta*(w/2+2)-1,sin_theta*(h+1)-1])
    off = npy.array([cos_theta*(w/2),sin_theta*(h/2)])
    dir = npy.array([cos_theta,sin_theta])*dashpad
    cw = c2 + off +dir

    self._mytext.set_position(transform.inverse_xy_tup(tuple(cw)))
    # Now set the window extent
    # I'm not at all sure this is the right way to do this.
    we = self._mytext.get_window_extent(renderer=renderer)
    self._window_extent = we.deepcopy()
    self._window_extent.update(((c1[0], c1[1]),), False)

    # Finally, make text align center
    self._mytext.set_horizontalalignment('center')
    self._mytext.set_verticalalignment('center')

def tick_update_position(tick, x,y,z, angle):
    #
    tick.tick1On = False
    tick.tick2On = False
    tick.tick1line.set_data((x, x),(y,y))
    tick.tick2line.set_data((x, x),(y,y))
    tick.gridline.set_data((x, x),(y,y))
    #
    tick.label1.set_dashlength(8)
    tick.label1.set_dashrotation(angle)
    tick.label1.set_position((x,y))
    tick.label2.set_position((x,y))

class Axis(axis.XAxis):
    def __init__(self, adir, v_intervalx, d_intervalx, axes, *args, **kwargs):
        # adir identifies which axes this is
        self.adir = adir
        # data and viewing intervals for this direction
        self.d_interval = d_intervalx
        self.v_interval = v_intervalx
        #
        axis.XAxis.__init__(self, axes, *args, **kwargs)
        self.line = lines.Line2D(xdata=(0,0),ydata=(0,0),
                                 linewidth=0.75,
                                 color=(0,0,0,0),
                                 antialiased=True,
                           )
        #
        # these are the panes which surround the boundary of the view
        self.pane_bg_color = (0.95,0.95,0.95,0.1)
        self.pane_fg_color = (0.9,0.9,0.9,0.5)
        self.pane = patches.Polygon([],
                                    alpha=0.2,
                                    facecolor=self.pane_fg_color,
                                    edgecolor=self.pane_fg_color)
        #
        self.axes._set_artist_props(self.line)
        self.axes._set_artist_props(self.pane)
        self.gridlines = art3d.Line3DCollection([])
        self.axes._set_artist_props(self.gridlines)
        self.axes._set_artist_props(self.label)
        self.label._transform = self.axes.transData

    def get_tick_positions(self):
        majorTicks = self.get_major_ticks()
        majorLocs = self.major.locator()
        self.major.formatter.set_locs(majorLocs)
        majorLabels = [self.major.formatter(val, i) for i, val in enumerate(majorLocs)]
        return majorLabels,majorLocs

    def get_major_ticks(self):
        ticks = axis.XAxis.get_major_ticks(self)
        for t in ticks:
            def update_coords(renderer,self=t.label1):
                return text_update_coords(self, renderer)
            # Text overrides setattr so need this to force new method
            #t.label1.__dict__['update_coords'] = update_coords
            t.tick1line.set_transform(self.axes.transData)
            t.tick2line.set_transform(self.axes.transData)
            t.gridline.set_transform(self.axes.transData)
            t.label1.set_transform(self.axes.transData)
            t.label2.set_transform(self.axes.transData)
        #
        return ticks

    def set_pane_fg(self, xys):
        self.pane.xy = xys
        self.pane.set_edgecolor(self.pane_fg_color)
        self.pane.set_facecolor(self.pane_fg_color)
        self.pane.set_alpha(self.pane_fg_color[-1])

    def set_pane_bg(self, xys):
        self.pane.xy = xys
        self.pane.set_edgecolor(self.pane_bg_color)
        self.pane.set_facecolor(self.pane_bg_color)
        self.pane.set_alpha(self.pane_bg_color[-1])

    def draw(self, renderer):
        #
        self.label._transform = self.axes.transData
        renderer.open_group('axis3d')
        ticklabelBoxes = []
        ticklabelBoxes2 = []

        # code from XAxis
        majorTicks = self.get_major_ticks()
        majorLocs = self.major.locator()
        self.major.formatter.set_locs(majorLocs)
        majorLabels = [self.major.formatter(val, i)
                       for i, val in enumerate(majorLocs)]
        #
        minx,maxx,miny,maxy,minz,maxz = self.axes.get_w_lims()

        interval = self.get_view_interval()
        # filter locations here so that no extra grid lines are drawn
        majorLocs = [loc for loc in majorLocs if interval.contains(loc)]
        # these will generate spacing for labels and ticks
        dx = (maxx-minx)/12
        dy = (maxy-miny)/12
        dz = (maxz-minz)/12

        # stretch the boundary slightly so that the ticks have a better fit
        minx,maxx,miny,maxy,minz,maxz = (
            minx-dx/4,maxx+dx/4,miny-dy/4,maxy+dy/4,minz-dz/4,maxz+dz/4)

        # generate the unit_cubes and transformed unit_cubes from the stretched
        # limits
        vals = minx,maxx,miny,maxy,minz,maxz
        uc = self.axes.unit_cube(vals)
        tc = self.axes.tunit_cube(vals,renderer.M)
        #
        # these are flags which decide whether the axis should be drawn
        # on the high side (ie on the high side of the paired axis)
        xhigh = tc[1][2]>tc[2][2]
        yhigh = tc[3][2]>tc[2][2]
        zhigh = tc[0][2]>tc[2][2]
        #
        aoff = 0

        # lx,ly,lz are the label positions in user coordinates
        # to and te are the locations of the origin and the end of the axis
        #
        if self.adir == 'x':
            lx = (minx+maxx)/2
            if xhigh:
                # xaxis at front
                self.set_pane_fg([tc[0],tc[1],tc[5],tc[4]])
                to = tc[3]
                te = tc[2]
                xyz = [(x,maxy,minz) for x in majorLocs]
                nxyz = [(x,miny,minz) for x in majorLocs]
                lxyz = [(x,miny,maxz) for x in majorLocs]
                aoff = -90

                ly = maxy + dy
                lz = minz - dz
            else:
                self.set_pane_bg([tc[3],tc[2],tc[6],tc[7]])
                to = tc[0]
                te = tc[1]
                xyz = [(x,miny,minz) for x in majorLocs]
                nxyz = [(x,maxy,minz) for x in majorLocs]
                lxyz = [(x,maxy,maxz) for x in majorLocs]
                aoff = 90

                ly = miny - dy
                lz = minz - dz

        elif self.adir == 'y':
            # cube 3 is minx,maxy,minz
            # cube 2 is maxx,maxy,minz
            ly = (maxy+miny)/2
            if yhigh:
                # yaxis at front
                self.set_pane_fg([tc[0],tc[3],tc[7],tc[4]])
                to = tc[1]
                te = tc[2]
                xyz = [(maxx,y,minz) for y in majorLocs]
                nxyz = [(minx,y,minz) for y in majorLocs]
                lxyz = [(minx,y,maxz) for y in majorLocs]
                aoff = 90

                #
                lx = maxx + dx
                lz = minz - dz

            else:
                # yaxis at back
                self.set_pane_bg([tc[1],tc[5],tc[6],tc[2]])
                to = tc[0]
                te = tc[3]
                xyz = [(minx,y,minz) for y in majorLocs]
                nxyz = [(maxx,y,minz) for y in majorLocs]
                lxyz = [(maxx,y,maxz) for y in majorLocs]
                aoff = -90
                #
                lx = minx - dx
                lz = minz - dz

        elif self.adir == 'z':
            nxyz = None
            self.set_pane_bg([tc[0],tc[1],tc[2],tc[3]])
            aoff = -90
            lz = (maxz+minz)/2
            if xhigh and yhigh:
                to = tc[1]
                te = tc[5]
                xyz = [(maxx,miny,z) for z in majorLocs]
                nxyz = [(minx,miny,z) for z in majorLocs]
                lxyz = [(minx,maxy,z) for z in majorLocs]
                #
                lx = maxx + dx
                ly = miny - dy
            elif xhigh and not yhigh:
                to = tc[2]
                te = tc[6]
                xyz = [(maxx,maxy,z) for z in majorLocs]
                nxyz = [(maxx,miny,z) for z in majorLocs]
                lxyz = [(minx,miny,z) for z in majorLocs]

                lx = maxx + dx
                ly = maxy + dy
            elif yhigh and not xhigh:
                to = tc[0]
                te = tc[4]
                xyz = [(minx,miny,z) for z in majorLocs]
                nxyz = [(minx,maxy,z) for z in majorLocs]
                lxyz = [(maxx,maxy,z) for z in majorLocs]
                lx = minx - dx
                ly = miny - dy
            else:
                to = tc[3]
                te = tc[7]
                xyz = [(minx,maxy,z) for z in majorLocs]
                nxyz = [(maxx,maxy,z) for z in majorLocs]
                lxyz = [(maxx,miny,z) for z in majorLocs]
                lx = minx - dx
                ly = maxy + dy

        #
        tlx,tly,tlz = proj3d.proj_transform(lx,ly,lz, renderer.M)
        self.label.set_position((tlx,tly))

        self.label.set_va('center')
        #print self.label._text, lx,ly, tlx,tly
        #
        self.pane.draw(renderer)
        #TODO - why didn't this work earlier ?
        self.pane.set_transform(self.axes.transData)
        self.gridlines.set_transform(self.axes.transData)
        #
        self.line.set_transform(self.axes.transData)
        self.line.set_data((to[0],te[0]),(to[1],te[1]))
        self.line.draw(renderer)

        angle = norm_angle(math.degrees(math.atan2(te[1]-to[1],te[0]-to[0])))
        #
        # should be some other enabler here...
        if len(self.label._text)>1:
            if abs(angle)>90 and self.adir != 'z':
                la = angle+180
            else:
                la = angle
            # almight kludge - the text angles seem to be incorrect
            # (at-least for gtkagg backend...)
            # this seems to more or less fix the problem...
            if 0:
                rla = math.radians(la)
                # -15 gives the closest result ... but the perspective projection is
                # then slightly broken..
                erra = -12*math.cos(rla)*math.sin(rla)
                self.label.set_rotation(la + erra)
            else:
                self.label.set_rotation(la)

        #
        self.label.draw(renderer)
        #
        angle = angle + aoff

        if xyz:
            points = proj3d.proj_points(xyz,renderer.M)
        if nxyz:
            tnxyz = proj3d.proj_points(nxyz,renderer.M)
            tlxyz = proj3d.proj_points(lxyz,renderer.M)
            lines = zip(xyz,nxyz,lxyz)
            self.gridlines.segments_3d = lines
            self.gridlines._colors = [(0.9,0.9,0.9,1)]*len(lines)
            #self.gridlines._colors = [(0.98,0.98,0.98,1.0)]*len(lines)
            self.gridlines.draw(renderer)

        if xyz:
            seen = {}
            interval = self.get_view_interval()
            for tick, loc, (x,y,z), label in zip(majorTicks,
                                                 majorLocs, points,
                                                 majorLabels):
                if tick is None: continue
                if not interval.contains(loc): continue
                seen[loc] = 1
                tick_update_position(tick, x,y,z, angle=angle)
                tick.set_label1(label)
                tick.set_label2(label)
                tick.draw(renderer)
                if tick.label1On:
                    extent = tick.label1.get_window_extent(renderer)
                    ticklabelBoxes.append(extent)
                if tick.label2On:
                    extent = tick.label2.get_window_extent(renderer)
                    ticklabelBoxes2.append(extent)
        #
        renderer.close_group('axis3d')

    def get_view_interval(self):
        """return the Interval instance for this axis view limits
        """
        return self.v_interval()

    def get_data_interval(self):
        'return the Interval instance for this axis data limits'
        return self.d_interval()
