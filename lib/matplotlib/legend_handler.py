import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
# from matplotlib.collections import LineCollection, RegularPolyCollection, \
#      CircleCollection

class Handler(object):
    def __init__(self, xpad=0., ypad=0., update_func=None):
        self._xpad, self._ypad = xpad, ypad
        self._update_prop_func = update_func
        
    def _update_prop(self, legend_handle, orig_handle):
        if self._update_prop_func is None:
            self._default_update_prop(legend_handle, orig_handle)
        else:
            self._update_prop_func(legend_handle, orig_handle)

    def _default_update_prop(self, legend_handle, orig_handle):
        legend_handle.update_from(orig_handle)

        
    def _update(self, legend_handle, orig_handle, legend):

        self._update_prop(legend_handle, orig_handle)

        legend._set_artist_props(legend_handle)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

        # make usre that transform is not set since they will be set
        # when added to an handlerbox.
        legend_handle._transformSet = False
        
    def adjust_drawing_area(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize,
                            ):
        xdescent = xdescent-self._xpad*fontsize
        ydescent = ydescent-self._ypad*fontsize
        width = width-self._xpad*fontsize
        height = height-self._ypad*fontsize
        return xdescent, ydescent, width, height

    def __call__(self, legend, orig_handle,
                 fontsize,
                 handlebox):
        """
        x, y, w, h in display coordinate w/ default dpi (72)
        fontsize in points
        """

        width, height, xdescent, ydescent = handlebox.width, \
                                            handlebox.height, \
                                            handlebox.xdescent, \
                                            handlebox.ydescent

        xdescent, ydescent, width, height = \
                  self.adjust_drawing_area(legend, orig_handle,
                                           xdescent, ydescent, width, height,
                                           fontsize)
        a_list = self.create_artists(legend, orig_handle,
                                     xdescent, ydescent, width, height, fontsize,
                                     handlebox.get_transform())

        # create_artists will return a list of artists.
        for a in a_list:
            handlebox.add_artist(a)
        
        # we only return the first artist
        return a_list[0]
    

class HandlerLine2D(Handler):
    def __init__(self, marker_pad=0.3, npoints=None, **kw):
        Handler.__init__(self, **kw)
        self._marker_pad = marker_pad
        self._npoints = None

    def get_npoints(self, legend):
        if self._npoints is None:
            return legend.numpoints
        else:
            return self._npoints
        
    def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
        npoints = self.get_npoints(legend)
        
        if npoints > 1:
            # we put some pad here to compensate the size of the
            # marker
            xdata = np.linspace(-xdescent+self._marker_pad*fontsize,
                                width-self._marker_pad*fontsize,
                                npoints)
            xdata_marker = xdata
        elif npoints == 1:
            xdata = np.linspace(-xdescent, width, 2)
            xdata_marker = [0.5*width-0.5*xdescent]

        return xdata, xdata_marker
    


    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = ((height-ydescent)/2.)*np.ones(xdata.shape, float)
        legline = Line2D(xdata, ydata)

        self._update(legline, orig_handle, legend)
        #legline.update_from(orig_handle)
        #legend._set_artist_props(legline) # after update
        #legline.set_clip_box(None)
        #legline.set_clip_path(None)
        legline.set_drawstyle('default')
        legline.set_marker('None')


        legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        self._update(legline_marker, orig_handle, legend)
        #legline_marker.update_from(orig_handle)
        #legend._set_artist_props(legline_marker)
        #legline_marker.set_clip_box(None)
        #legline_marker.set_clip_path(None)
        legline_marker.set_linestyle('None')
        if legend.markerscale !=1:
            newsz = legline_marker.get_markersize()*legend.markerscale
            legline_marker.set_markersize(newsz)
        # we don't want to add this to the return list because
        # the texts and handles are assumed to be in one-to-one
        # correpondence.
        legline._legmarker = legline_marker
        
        return [legline, legline_marker]
    


class HandlerPatch(Handler):
    def __init__(self, patch_func=None, **kw):
        Handler.__init__(self, **kw)

        self._patch_func = patch_func

    def _create_patch(self, legend, orig_handle,
                      xdescent, ydescent, width, height, fontsize):
        if self._patch_func is None:
            p = Rectangle(xy=(-xdescent, -ydescent),
                          width = width+xdescent, height=(height+ydescent))
        else:
            p = self._patch_func(legend=legend, orig_handle=orig_handle,
                                 xdescent=xdescent, ydescent=ydescent,
                                 width=width, height=height, fontsize=fontsize)

        return p

    # def _update(self, legend_handle, orig_handle, legend):

    #     self._update_prop(legend_handle, orig_handle)

    #     legend._set_artist_props(legend_handle)
    #     legend_handle.set_clip_box(None)
    #     legend_handle.set_clip_path(None)
        
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):

        p = self._create_patch(legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize)

        self._update(p, orig_handle, legend)

        return [p]



class HandlerLineCollection(HandlerLine2D):

    def get_npoints(self, legend):
        if self._npoints is None:
            return legend.scatterpoints
        else:
            return self._npoints

    def _default_update_prop(self, legend_handle, orig_handle):
        lw = orig_handle.get_linewidth()[0]
        dashes = orig_handle.get_dashes()[0]
        color = orig_handle.get_colors()[0]
        legend_handle.set_color(color)
        legend_handle.set_linewidth(lw)
        if dashes[0] is not None: # dashed line
            legend_handle.set_dashes(dashes[1])


    # def _update(self, legend_handle, orig_handle, legend):

    #     self._update_prop(legend_handle, orig_handle)

    #     legend._set_artist_props(legend_handle)
    #     legend_handle.set_clip_box(None)
    #     legend_handle.set_clip_path(None)

    #     # legend._set_artist_props(legend_handle)
    #     # legend_handle.set_clip_box(None)
    #     # legend_handle.set_clip_path(None)
        

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = ((height-ydescent)/2.)*np.ones(xdata.shape, float)
        legline = Line2D(xdata, ydata)

        self._update(legline, orig_handle, legend)

        return [legline]



class HandlerRegularPolyCollection(HandlerLine2D):
    def __init__(self, scatteryoffsets=None, sizes=None, **kw):
        HandlerLine2D.__init__(self, **kw)

        self._scatteryoffsets = scatteryoffsets
        self._sizes = sizes
        
    def get_npoints(self, legend):
        if self._npoints is None:
            return legend.scatterpoints
        else:
            return self._npoints

    def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
        if self._scatteryoffsets is None:
            ydata = height*legend._scatteryoffsets
        else:
            ydata = height*np.asarray(self._scatteryoffsets)

        return ydata

    def get_sizes(self, legend, orig_handle,
                 xdescent, ydescent, width, height, fontsize):
        if self._sizes is None:
            size_max = max(orig_handle.get_sizes())*legend.markerscale**2
            size_min = min(orig_handle.get_sizes())*legend.markerscale**2

            npoints = self.get_npoints(legend)
            if npoints < 4:
                sizes = [.5*(size_max+size_min), size_max,
                         size_min]
            else:
                sizes = (size_max-size_min)*np.linspace(0,1,npoints)+size_min
        else:
            sizes = self._sizes #[:legend.scatterpoints]

        return sizes
        
    def _update(self, legend_handle, orig_handle, legend):

        self._update_prop(legend_handle, orig_handle)

        legend_handle.set_figure(legend.figure)
        #legend._set_artist_props(legend_handle)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)(orig_handle.get_numsides(),
                              rotation=orig_handle.get_rotation(),
                              sizes=sizes,
                              offsets=offsets,
                              transOffset=transOffset,
                              )
        return p
    
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):


        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)


        ydata = self.get_ydata(legend, xdescent, ydescent,
                               width, height, fontsize)
        
        sizes = self.get_sizes(legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize)
        
        p = self.create_collection(orig_handle, sizes,
                                   offsets=zip(xdata_marker,ydata),
                                   transOffset=trans)

        self._update(p, orig_handle, legend)

        p._transOffset = trans
        p.set_transform(None)

        return [p]


class HandlerCircleCollection(HandlerRegularPolyCollection):
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)(sizes,
                              offsets=offsets,
                              transOffset=transOffset,
                              )
        return p


class HandlerMulti(Handler):
    def __init__(self, *handle_list, **kwargs):
        Handler.__init__(self, **kwargs)

        self._handle_list = handle_list

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        handler_map = legend.get_legend_handler_map()
        a_list = []
        for handle1 in self._handle_list:
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(legend, handle1,
                                             xdescent, ydescent, width, height,
                                             fontsize,
                                             trans)
            a_list.extend(_a_list)

        return a_list
