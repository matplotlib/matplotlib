"""
This module defines default legend handlers.

Legend handlers are expected to be a callable object with a following
signature. ::

    legend_handler(legend, orig_handle, fontsize, handlebox)

Where *legend* is the legend itself, *orig_handle* is the original
plot, *fontsize* is the fontsize in pixles, and *handlebox* is a
OffsetBox instance. Within the call, you should create relevant
artists (using relevant properties from the *legend* and/or
*orig_handle*) and add them into the handlebox. The artists needs to
be scaled according to the fontsize (note that the size is in pixel,
i.e., this is dpi-scaled value).

This module includes definition of several legend handler classes
derived from the base class (HandlerBase) with a following method.

    def __call__(self, legend, orig_handle,
                 fontsize,
                 handlebox):


"""

import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll


def update_from_first_child(tgt, src):
    tgt.update_from(src.get_children()[0])


class HandlerBase(object):
    """
    A Base class for default legend handlers.

    The derived classes are meant to override *create_artists* method, which
    has a following signature.::

      def create_artists(self, legend, orig_handle,
                         xdescent, ydescent, width, height, fontsize,
                         trans):

    The overridden method needs to create artists of the given
    transform that fits in the given dimension (xdescent, ydescent,
    width, height) that are scaled by fontsize if necessary.

    """
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

    def update_prop(self, legend_handle, orig_handle, legend):

        self._update_prop(legend_handle, orig_handle)

        legend._set_artist_props(legend_handle)
        legend_handle.set_clip_box(None)
        legend_handle.set_clip_path(None)

    def adjust_drawing_area(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize,
                            ):
        xdescent = xdescent - self._xpad * fontsize
        ydescent = ydescent - self._ypad * fontsize
        width = width - self._xpad * fontsize
        height = height - self._ypad * fontsize
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

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        raise NotImplementedError('Derived must override')


class HandlerNpoints(HandlerBase):
    def __init__(self, marker_pad=0.3, numpoints=None, **kw):
        HandlerBase.__init__(self, **kw)

        self._numpoints = numpoints
        self._marker_pad = marker_pad

    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.numpoints
        else:
            return self._numpoints

    def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
        numpoints = self.get_numpoints(legend)

        if numpoints > 1:
            # we put some pad here to compensate the size of the
            # marker
            xdata = np.linspace(-xdescent + self._marker_pad * fontsize,
                                width - self._marker_pad * fontsize,
                                numpoints)
            xdata_marker = xdata
        elif numpoints == 1:
            xdata = np.linspace(-xdescent, width, 2)
            xdata_marker = [0.5 * width - 0.5 * xdescent]

        return xdata, xdata_marker


class HandlerNpointsYoffsets(HandlerNpoints):
    def __init__(self, numpoints=None, yoffsets=None, **kw):
        HandlerNpoints.__init__(self, numpoints=numpoints, **kw)
        self._yoffsets = yoffsets

    def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
        if self._yoffsets is None:
            ydata = height * legend._scatteryoffsets
        else:
            ydata = height * np.asarray(self._yoffsets)

        return ydata


class HandlerLine2D(HandlerNpoints):
    """
    Handler for Line2D instances
    """
    def __init__(self, marker_pad=0.3, numpoints=None, **kw):
        HandlerNpoints.__init__(self, marker_pad=marker_pad, numpoints=numpoints, **kw)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
        legline = Line2D(xdata, ydata)

        self.update_prop(legline, orig_handle, legend)
        legline.set_drawstyle('default')
        legline.set_marker("")

        legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        self.update_prop(legline_marker, orig_handle, legend)
        legline_marker.set_linestyle('None')
        if legend.markerscale != 1:
            newsz = legline_marker.get_markersize() * legend.markerscale
            legline_marker.set_markersize(newsz)
        # we don't want to add this to the return list because
        # the texts and handles are assumed to be in one-to-one
        # correspondence.
        legline._legmarker = legline_marker

        legline.set_transform(trans)
        legline_marker.set_transform(trans)

        return [legline, legline_marker]

class HandlerPatch(HandlerBase):
    """
    Handler for Patches
    """
    def __init__(self, patch_func=None, **kw):
        HandlerBase.__init__(self, **kw)

        self._patch_func = patch_func

    def _create_patch(self, legend, orig_handle,
                      xdescent, ydescent, width, height, fontsize):
        if self._patch_func is None:
            p = Rectangle(xy=(-xdescent, -ydescent),
                          width=width, height=height)
        else:
            p = self._patch_func(legend=legend, orig_handle=orig_handle,
                                 xdescent=xdescent, ydescent=ydescent,
                                 width=width, height=height, fontsize=fontsize)
        return p

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):

        p = self._create_patch(legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize)

        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerLineCollection(HandlerLine2D):
    """
    Handler for LineCollections
    """
    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints

    def _default_update_prop(self, legend_handle, orig_handle):
        lw = orig_handle.get_linewidth()[0]
        dashes = orig_handle.get_dashes()[0]
        color = orig_handle.get_colors()[0]
        legend_handle.set_color(color)
        legend_handle.set_linewidth(lw)
        if dashes[0] is not None: # dashed line
            legend_handle.set_dashes(dashes[1])

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
        legline = Line2D(xdata, ydata)

        self.update_prop(legline, orig_handle, legend)
        legline.set_transform(trans)

        return [legline]


class HandlerRegularPolyCollection(HandlerNpointsYoffsets):
    """
    Handler for RegularPolyCollections.
    """
    def __init__(self, yoffsets=None, sizes=None, **kw):
        HandlerNpointsYoffsets.__init__(self, yoffsets=yoffsets, **kw)

        self._sizes = sizes

    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints

    def get_sizes(self, legend, orig_handle,
                 xdescent, ydescent, width, height, fontsize):
        if self._sizes is None:
            size_max = max(orig_handle.get_sizes()) * legend.markerscale ** 2
            size_min = min(orig_handle.get_sizes()) * legend.markerscale ** 2

            numpoints = self.get_numpoints(legend)
            if numpoints < 4:
                sizes = [.5 * (size_max + size_min), size_max,
                         size_min]
            else:
                rng = (size_max - size_min)
                sizes = rng * np.linspace(0, 1, numpoints) + size_min
        else:
            sizes = self._sizes

        return sizes

    def update_prop(self, legend_handle, orig_handle, legend):

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
                                   offsets=zip(xdata_marker, ydata),
                                   transOffset=trans)

        self.update_prop(p, orig_handle, legend)
        p._transOffset = trans
        return [p]


class HandlerPathCollection(HandlerRegularPolyCollection):
    """
    Handler for PathCollections, which are used by scatter
    """
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)([orig_handle.get_paths()[0]],
                              sizes=sizes,
                              offsets=offsets,
                              transOffset=transOffset,
                              )
        return p


class HandlerCircleCollection(HandlerRegularPolyCollection):
    """
    Handler for CircleCollections
    """
    def create_collection(self, orig_handle, sizes, offsets, transOffset):
        p = type(orig_handle)(sizes,
                              offsets=offsets,
                              transOffset=transOffset,
                              )
        return p


class HandlerErrorbar(HandlerLine2D):
    """
    Handler for Errorbars
    """
    def __init__(self, xerr_size=0.5, yerr_size=None,
                 marker_pad=0.3, numpoints=None, **kw):

        self._xerr_size = xerr_size
        self._yerr_size = yerr_size

        HandlerLine2D.__init__(self, marker_pad=marker_pad, numpoints=numpoints,
                               **kw)

    def get_err_size(self, legend, xdescent, ydescent, width, height, fontsize):
        xerr_size = self._xerr_size * fontsize

        if self._yerr_size is None:
            yerr_size = xerr_size
        else:
            yerr_size = self._yerr_size * fontsize

        return xerr_size, yerr_size

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        plotlines, caplines, barlinecols = orig_handle

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
        legline = Line2D(xdata, ydata)


        xdata_marker = np.asarray(xdata_marker)
        ydata_marker = np.asarray(ydata[:len(xdata_marker)])

        xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent,
                                                 width, height, fontsize)

        legline_marker = Line2D(xdata_marker, ydata_marker)

        # when plotlines are None (only errorbars are drawn), we just
        # make legline invisible.
        if plotlines is None:
            legline.set_visible(False)
            legline_marker.set_visible(False)
        else:
            self.update_prop(legline, plotlines, legend)

            legline.set_drawstyle('default')
            legline.set_marker('None')

            self.update_prop(legline_marker, plotlines, legend)
            legline_marker.set_linestyle('None')

            if legend.markerscale != 1:
                newsz = legline_marker.get_markersize() * legend.markerscale
                legline_marker.set_markersize(newsz)

        handle_barlinecols = []
        handle_caplines = []

        if orig_handle.has_xerr:
            verts = [ ((x - xerr_size, y), (x + xerr_size, y))
                      for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)

            if caplines:
                capline_left = Line2D(xdata_marker - xerr_size, ydata_marker)
                capline_right = Line2D(xdata_marker + xerr_size, ydata_marker)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker("|")
                capline_right.set_marker("|")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        if orig_handle.has_yerr:
            verts = [ ((x, y - yerr_size), (x, y + yerr_size))
                      for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)

            if caplines:
                capline_left = Line2D(xdata_marker, ydata_marker - yerr_size)
                capline_right = Line2D(xdata_marker, ydata_marker + yerr_size)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker("_")
                capline_right.set_marker("_")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        artists = []
        artists.extend(handle_barlinecols)
        artists.extend(handle_caplines)
        artists.append(legline)
        artists.append(legline_marker)

        for artist in artists:
            artist.set_transform(trans)

        return artists

class HandlerStem(HandlerNpointsYoffsets):
    """
    Handler for Errorbars
    """
    def __init__(self, marker_pad=0.3, numpoints=None,
                 bottom=None, yoffsets=None, **kw):

        HandlerNpointsYoffsets.__init__(self, marker_pad=marker_pad,
                                        numpoints=numpoints,
                                        yoffsets=yoffsets,
                                        **kw)
        self._bottom = bottom

    def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
        if self._yoffsets is None:
            ydata = height * (0.5 * legend._scatteryoffsets + 0.5)
        else:
            ydata = height * np.asarray(self._yoffsets)

        return ydata

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        markerline, stemlines, baseline = orig_handle

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = self.get_ydata(legend, xdescent, ydescent,
                               width, height, fontsize)

        if self._bottom is None:
            bottom = 0.
        else:
            bottom = self._bottom

        leg_markerline = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        self.update_prop(leg_markerline, markerline, legend)

        leg_stemlines = []
        for thisx, thisy in zip(xdata_marker, ydata):
            l = Line2D([thisx, thisx], [bottom, thisy])
            leg_stemlines.append(l)

        for lm, m in zip(leg_stemlines, stemlines):
            self.update_prop(lm, m, legend)

        leg_baseline = Line2D([np.amin(xdata), np.amax(xdata)],
                              [bottom, bottom])

        self.update_prop(leg_baseline, baseline, legend)

        artists = [leg_markerline]
        artists.extend(leg_stemlines)
        artists.append(leg_baseline)

        for artist in artists:
            artist.set_transform(trans)

        return artists


class HandlerTuple(HandlerBase):
    """
    Handler for Tuple
    """
    def __init__(self, **kwargs):
        HandlerBase.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        handler_map = legend.get_legend_handler_map()
        a_list = []
        for handle1 in orig_handle:
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(legend, handle1,
                                             xdescent, ydescent, width, height,
                                             fontsize,
                                             trans)
            a_list.extend(_a_list)

        return a_list
