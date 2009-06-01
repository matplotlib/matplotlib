from __future__ import division

import matplotlib
rcParams = matplotlib.rcParams

import matplotlib.artist as martist
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import warnings

class Spine(martist.Artist):
    """an axis spine -- the line noting the data area boundaries

    Spines are the lines connecting the axis tick marks and noting the
    boundaries of the data area. They can be placed at arbitrary
    positions. See function:`~matplotlib.spines.Spine.set_position`
    for more information.

    The default position is ``('outward',0)``.
    """
    def __str__(self):
        return "Spine"

    def __init__(self,axes,spine_type,artist):
        """
        - *axes* : the Axes instance containing the spine
        - *spine_type* : a string specifying the spine type
        - *artist* : the artist instance used to draw the spine
        """
        martist.Artist.__init__(self)
        self.axes = axes
        self.set_figure(self.axes.figure)
        self.spine_type = spine_type
        self.artist = artist
        self.color = rcParams['axes.edgecolor']
        self.axis = None

        if isinstance(self.artist,mlines.Line2D):
            self.artist.set_color(self.color)
            self.artist.set_linewidth(rcParams['axes.linewidth'])
        elif isinstance(self.artist,mpatches.Patch):
            self.artist.set_facecolor('none')
            self.artist.set_edgecolor(self.color)
            self.artist.set_linewidth(rcParams['axes.linewidth'])
        self.artist.set_zorder(2.5)
        self.artist.set_transform(self.axes.transAxes) # default transform

        # Defer initial position determination. (Not much support for
        # non-rectangular axes is currently implemented, and this lets
        # them pass through the spines machinery without errors.)
        self._position = None

    def _ensure_position_is_set(self):
        if self._position is None:
            # default position
            self._position = ('outward',0.0) # in points
            self.set_position(self._position)

    def register_axis(self,axis):
        """register an axis

        An axis should be registered with its corresponding spine from
        the Axes instance. This allows the spine to clear any axis
        properties when needed.
        """
        self.axis = axis
        if self.axis is not None:
            self.axis.cla()

    def cla(self):
        'Clear the current spine'
        self._position = None # clear position
        if self.axis is not None:
            self.axis.cla()

    @allow_rasterization
    def draw(self,renderer):
        "draw everything that belongs to the spine"
        if self.color=='none':
            # don't draw invisible spines
            return
        self.artist.draw(renderer)

    def _calc_offset_transform(self):
        """calculate the offset transform performed by the spine"""
        self._ensure_position_is_set()
        position = self._position
        if isinstance(position,basestring):
            if position=='center':
                position = ('axes',0.5)
            elif position=='zero':
                position = ('data',0)
        assert len(position)==2, "position should be 2-tuple"
        position_type, amount = position
        assert position_type in ('axes','outward','data')
        if position_type=='outward':
            if amount == 0:
                # short circuit commonest case
                self._spine_transform =  ('identity',mtransforms.IdentityTransform())
            elif self.spine_type in ['left','right','top','bottom']:
                offset_vec = {'left':(-1,0),
                              'right':(1,0),
                              'bottom':(0,-1),
                              'top':(0,1),
                              }[self.spine_type]
                # calculate x and y offset in dots
                offset_x = amount*offset_vec[0]/ 72.0
                offset_y = amount*offset_vec[1]/ 72.0
                self._spine_transform = ('post',
                                         mtransforms.ScaledTranslation(offset_x,offset_y,
                                                                       self.figure.dpi_scale_trans))
            else:
                warnings.warn('unknown spine type "%s": no spine '
                              'offset performed'%self.spine_type)
                self._spine_transform = ('identity',mtransforms.IdentityTransform())
        elif position_type=='axes':
            if self.spine_type in ('left','right'):
                self._spine_transform = ('pre',
                                         mtransforms.Affine2D().translate(amount, 0.0))
            elif self.spine_type in  ('bottom','top'):
                self._spine_transform = ('pre',
                                         mtransforms.Affine2D().translate(0.0, amount))
            else:
                warnings.warn('unknown spine type "%s": no spine '
                              'offset performed'%self.spine_type)
                self._spine_transform = ('identity',mtransforms.IdentityTransform())
        elif position_type=='data':
            if self.spine_type in ('left','right'):
                self._spine_transform = ('data',
                                         mtransforms.Affine2D().translate(amount,0))
            elif self.spine_type in ('bottom','top'):
                self._spine_transform = ('data',
                                         mtransforms.Affine2D().translate(0,amount))
            else:
                warnings.warn('unknown spine type "%s": no spine '
                              'offset performed'%self.spine_type)
                self._spine_transform =  ('identity',mtransforms.IdentityTransform())

    def set_position(self,position):
        """set the position of the spine

        Spine position is specified by a 2 tuple of (position type,
        amount). The position types are:

        * 'outward' : place the spine out from the data area by the
          specified number of points. (Negative values specify placing the
          spine inward.)

        * 'axes' : place the spine at the specified Axes coordinate (from
          0.0-1.0).

        * 'data' : place the spine at the specified data coordinate.

        Additionally, shorthand notations define a special positions:

        * 'center' -> ('axes',0.5)
        * 'zero' -> ('data', 0.0)

        """
        if position in ('center','zero'):
            # special positions
            pass
        else:
            assert len(position)==2, "position should be 'center' or 2-tuple"
            assert position[0] in ['outward','axes','data']
        self._position = position
        self._calc_offset_transform()

        t = self.get_spine_transform()
        if self.spine_type in ['left','right']:
            t2 = mtransforms.blended_transform_factory(t,
                                                       self.axes.transAxes)
        elif self.spine_type in ['bottom','top']:
            t2 = mtransforms.blended_transform_factory(self.axes.transAxes,
                                                       t)
        self.artist.set_transform(t2)

        if self.axis is not None:
            self.axis.cla()

    def get_position(self):
        """get the spine position"""
        self._ensure_position_is_set()
        return self._position

    def get_spine_transform(self):
        """get the spine transform"""
        self._ensure_position_is_set()
        what, how = self._spine_transform

        if what == 'data':
            # special case data based spine locations
            if self.spine_type in ['left','right']:
                data_xform = self.axes.transScale + \
                             (how+self.axes.transLimits + self.axes.transAxes)
                result = mtransforms.blended_transform_factory(
                    data_xform,self.axes.transData)
            elif self.spine_type in ['top','bottom']:
                data_xform = self.axes.transScale + \
                             (how+self.axes.transLimits + self.axes.transAxes)
                result = mtransforms.blended_transform_factory(
                    self.axes.transData,data_xform)
            else:
                raise ValueError('unknown spine spine_type: %s'%self.spine_type)
            return result

        if self.spine_type in ['left','right']:
            base_transform = self.axes.get_yaxis_transform(which='grid')
        elif self.spine_type in ['top','bottom']:
            base_transform = self.axes.get_xaxis_transform(which='grid')
        else:
            raise ValueError('unknown spine spine_type: %s'%self.spine_type)

        if what=='identity':
            return base_transform
        elif what=='post':
            return base_transform+how
        elif what=='pre':
            return how+base_transform
        else:
            raise ValueError("unknown spine_transform type: %s"%what)

    def set_color(self,value):
        """set the color of the spine artist

        Note: a value of 'none' will cause the artist not to be drawn.
        """
        self.color = value
        if isinstance(self.artist,mlines.Line2D):
            self.artist.set_color(self.color)
        elif isinstance(self.artist,mpatches.Patch):
            self.artist.set_edgecolor(self.color)

    def get_color(self):
        """get the color of the spine artist"""
        return self.color
