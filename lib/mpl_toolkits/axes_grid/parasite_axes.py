import warnings

import matplotlib
rcParams = matplotlib.rcParams
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcoll
import matplotlib.legend as mlegend

from matplotlib.axes import subplot_class_factory
from axislines import Axes

import numpy as np

import matplotlib.cbook as cbook
is_string_like = cbook.is_string_like


class ParasiteAxes(Axes):

    def __init__(self, parent_axes, **kargs):

        self._parent_axes = parent_axes
        kargs.update(dict(frameon=False))
        super(ParasiteAxes, self).__init__(parent_axes.figure,
                                           parent_axes._position, **kargs)


    #def apply_aspect(self, position=None):
    #    pos = self._parent_axes.get_position(original=False)
    #    self.set_position(pos, "active")


    def cla(self):
        super(ParasiteAxes, self).cla()

        martist.setp(self.get_children(), visible=False)
        self._get_lines = self._parent_axes._get_lines


    def get_images_artists(self):
        artists = set([a for a in self.get_children() if a.get_visible()])
        images = set([a for a in self.images if a.get_visible()])

        return list(images), list(artists - images)



class ParasiteAxesAuxTrans(ParasiteAxes):

    def __init__(self, parent_axes, aux_transform, viewlim_mode=None,
                 **kwargs):

        self.transAux = aux_transform

        #self._viewlim_mode = viewlim_mode
        self.set_viewlim_mode(viewlim_mode)

        super(ParasiteAxesAuxTrans, self).__init__(parent_axes, **kwargs)

    def _set_lim_and_transforms(self):

        self.transAxes = self._parent_axes.transAxes

        self.transData = \
            self.transAux + \
            self._parent_axes.transData

        self._xaxis_transform = mtransforms.blended_transform_factory(
                self.transData, self.transAxes)
        self._yaxis_transform = mtransforms.blended_transform_factory(
                self.transAxes, self.transData)

    def set_viewlim_mode(self, mode):
        if mode not in [None, "equal", "transform"]:
            raise ValueError("Unknown mode : %s" % (mode,))
        else:
            self._viewlim_mode = mode

    def get_viewlim_mode(self):
        return self._viewlim_mode


    def update_viewlim(self):
        viewlim = self._parent_axes.viewLim.frozen()
        mode = self.get_viewlim_mode()
        if mode is None:
            pass
        elif mode == "equal":
            self.axes.viewLim.set(viewlim)
        elif mode == "transform":
            self.axes.viewLim.set(viewlim.transformed(self.transAux.inverted()))
        else:
            raise ValueError("Unknown mode : %s" % (self._viewlim_mode,))


    def apply_aspect(self, position=None):
        self.update_viewlim()
        super(ParasiteAxesAuxTrans, self).apply_aspect()



    def _pcolor(self, method_name, *XYC, **kwargs):
        if len(XYC) == 1:
            C = XYC[0]
            ny, nx = C.shape

            gx = np.arange(-0.5, nx, 1.)
            gy = np.arange(-0.5, ny, 1.)

            X, Y = np.meshgrid(gx, gy)
        else:
            X, Y, C = XYC

        pcolor_routine = getattr(ParasiteAxes, method_name)

        if kwargs.has_key("transform"):
            mesh = pcolor_routine(self, X, Y, C, **kwargs)
        else:
            orig_shape = X.shape
            xy = np.vstack([X.flat, Y.flat])
            xyt=xy.transpose()
            wxy = self.transAux.transform(xyt)
            gx, gy = wxy[:,0].reshape(orig_shape), wxy[:,1].reshape(orig_shape)
            mesh = pcolor_routine(self, gx, gy, C, **kwargs)
            mesh.set_transform(self._parent_axes.transData)

        return mesh

    def pcolormesh(self, *XYC, **kwargs):
        return self._pcolor("pcolormesh", *XYC, **kwargs)

    def pcolor(self, *XYC, **kwargs):
        return self._pcolor("pcolor", *XYC, **kwargs)

    def _contour(self, method_name, *XYCL, **kwargs):

        if len(XYCL) <= 2:
            C = XYCL[0]
            ny, nx = C.shape

            gx = np.arange(0., nx, 1.)
            gy = np.arange(0., ny, 1.)

            X,Y = np.meshgrid(gx, gy)
            CL = XYCL
        else:
            X, Y = XYCL[:2]
            CL = XYCL[2:]

        contour_routine = getattr(ParasiteAxes, method_name)

        if kwargs.has_key("transform"):
            cont = contour_routine(self, X, Y, *CL, **kwargs)
        else:
            orig_shape = X.shape
            xy = np.vstack([X.flat, Y.flat])
            xyt=xy.transpose()
            wxy = self.transAux.transform(xyt)
            gx, gy = wxy[:,0].reshape(orig_shape), wxy[:,1].reshape(orig_shape)
            cont = contour_routine(self, gx, gy, *CL, **kwargs)
            for c in cont.collections:
                c.set_transform(self._parent_axes.transData)

        return cont

    def contour(self, *XYCL, **kwargs):
        return self._contour("contour", *XYCL, **kwargs)

    def contourf(self, *XYCL, **kwargs):
        return self._contour("contourf", *XYCL, **kwargs)



def _get_handles(ax):
    handles = ax.lines[:]
    handles.extend(ax.patches)
    handles.extend([c for c in ax.collections
                    if isinstance(c, mcoll.LineCollection)])
    handles.extend([c for c in ax.collections
                    if isinstance(c, mcoll.RegularPolyCollection)])
    return handles


class HostAxes(Axes):

    def __init__(self, *kl, **kwargs):

        self.parasites = []
        super(HostAxes, self).__init__(*kl, **kwargs)



    def legend(self, *args, **kwargs):

        if len(args)==0:
            all_handles = _get_handles(self)
            for ax in self.parasites:
                all_handles.extend(_get_handles(ax))
            handles = []
            labels = []
            for handle in all_handles:
                label = handle.get_label()
                if (label is not None and
                    label != '' and not label.startswith('_')):
                    handles.append(handle)
                    labels.append(label)
            if len(handles) == 0:
                warnings.warn("No labeled objects found. "
                              "Use label='...' kwarg on individual plots.")
                return None

        elif len(args)==1:
            # LABELS
            labels = args[0]
            handles = [h for h, label in zip(all_handles, labels)]

        elif len(args)==2:
            if is_string_like(args[1]) or isinstance(args[1], int):
                # LABELS, LOC
                labels, loc = args
                handles = [h for h, label in zip(all_handles, labels)]
                kwargs['loc'] = loc
            else:
                # LINES, LABELS
                handles, labels = args

        elif len(args)==3:
            # LINES, LABELS, LOC
            handles, labels, loc = args
            kwargs['loc'] = loc
        else:
            raise TypeError('Invalid arguments to legend')


        handles = cbook.flatten(handles)
        self.legend_ = mlegend.Legend(self, handles, labels, **kwargs)
        return self.legend_


    def draw(self, renderer):

        orig_artists = list(self.artists)
        orig_images = list(self.images)

        if hasattr(self, "get_axes_locator"):
            locator = self.get_axes_locator()
            if locator:
                pos = locator(self, renderer)
                self.set_position(pos, which="active")
                self.apply_aspect(pos)
            else:
                self.apply_aspect()
        else:
            self.apply_aspect()

        rect = self.get_position()

        for ax in self.parasites:
            ax.apply_aspect(rect)
            images, artists = ax.get_images_artists()
            self.images.extend(images)
            self.artists.extend(artists)

        super(HostAxes, self).draw(renderer)
        self.artists = orig_artists
        self.images = orig_images

    def cla(self):

        for ax in self.parasites:
            ax.cla()

        super(HostAxes, self).cla()


    def twinx(self):
        """
        call signature::

          ax2 = ax.twinx()

        create a twin of Axes for generating a plot with a sharex
        x-axis but independent y axis.  The y-axis of self will have
        ticks on left and the returned axes will have ticks on the
        right
        """

        ax2 = ParasiteAxes(self, sharex=self, frameon=False)
        self.parasites.append(ax2)

        # for normal axes
        self.yaxis.tick_left()
        ax2.xaxis.set_visible(False)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')

        # for axisline axes
        self._axislines["right"].set_visible(False)
        ax2._axislines["left"].set_visible(False)
        ax2._axislines["right"].set_visible(True)
        ax2._axislines["right"].major_ticklabels.set_visible(True)
        ax2._axislines["right"].label.set_visible(True)


        return ax2

    def twiny(self):
        """
        call signature::

          ax2 = ax.twiny()

        create a twin of Axes for generating a plot with a shared
        y-axis but independent x axis.  The x-axis of self will have
        ticks on bottom and the returned axes will have ticks on the
        top
        """

        ax2 = ParasiteAxes(self, sharey=self, frameon=False)
        self.parasites.append(ax2)

        # for normal axes
        self.xaxis.tick_bottom()
        ax2.yaxis.set_visible(False)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')

        # for axisline axes
        self._axislines["top"].set_visible(False)
        ax2._axislines["bottom"].set_visible(False)
        ax2._axislines["top"].set_visible(True)
        ax2._axislines["top"].major_ticklabels.set_visible(True)
        ax2._axislines["top"].label.set_visible(True)

        return ax2

    def twin(self, aux_trans=None):
        """
        call signature::

          ax2 = ax.twin()

        create a twin of Axes for generating a plot with a sharex
        x-axis but independent y axis.  The y-axis of self will have
        ticks on left and the returned axes will have ticks on the
        right
        """

        if aux_trans is None:
            ax2 = ParasiteAxesAuxTrans(self, mtransforms.IdentityTransform(),
                                       viewlim_mode="equal",
                                       )
        else:
            ax2 = ParasiteAxesAuxTrans(self, aux_trans,
                                       viewlim_mode="transform",
                                       )
        self.parasites.append(ax2)


        # for normal axes
        self.yaxis.tick_left()
        self.xaxis.tick_bottom()
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')

        # for axisline axes
        self._axislines["right"].set_visible(False)
        self._axislines["top"].set_visible(False)
        ax2._axislines["left"].set_visible(False)
        ax2._axislines["bottom"].set_visible(False)

        ax2._axislines["right"].set_visible(True)
        ax2._axislines["top"].set_visible(True)
        ax2._axislines["right"].major_ticklabels.set_visible(True)
        ax2._axislines["top"].major_ticklabels.set_visible(True)

        return ax2


SubplotHost = subplot_class_factory(HostAxes)


