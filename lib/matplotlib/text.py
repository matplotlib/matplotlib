"""
Classes for including text in a figure.
"""
from __future__ import division
import math

import numpy as np

from matplotlib import cbook
from matplotlib import rcParams
import artist
from artist import Artist
from cbook import is_string_like, maxdict
from font_manager import FontProperties
from patches import bbox_artist, YAArrow, FancyBboxPatch, \
     FancyArrowPatch, Rectangle
import transforms as mtransforms
from transforms import Affine2D, Bbox
from lines import Line2D

import matplotlib.nxutils as nxutils

def _process_text_args(override, fontdict=None, **kwargs):
    "Return an override dict.  See :func:`~pyplot.text' docstring for info"

    if fontdict is not None:
        override.update(fontdict)

    override.update(kwargs)
    return override

# Extracted from Text's method to serve as a function
def get_rotation(rotation):
    """
    Return the text angle as float.

    *rotation* may be 'horizontal', 'vertical', or a numeric value in degrees.
    """
    if rotation in ('horizontal', None):
        angle = 0.
    elif rotation == 'vertical':
        angle = 90.
    else:
        angle = float(rotation)
    return angle%360

# these are not available for the object inspector until after the
# class is build so we define an initial set here for the init
# function and they will be overridden after object defn
artist.kwdocd['Text'] =  """
    ========================== =========================================================================
    Property                   Value
    ========================== =========================================================================
    alpha                      float
    animated                   [True | False]
    backgroundcolor            any matplotlib color
    bbox                       rectangle prop dict plus key 'pad' which is a pad in points
    clip_box                   a matplotlib.transform.Bbox instance
    clip_on                    [True | False]
    color                      any matplotlib color
    family                     [ 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
    figure                     a matplotlib.figure.Figure instance
    fontproperties             a matplotlib.font_manager.FontProperties instance
    horizontalalignment or ha  [ 'center' | 'right' | 'left' ]
    label                      any string
    linespacing                float
    lod                        [True | False]
    multialignment             ['left' | 'right' | 'center' ]
    name or fontname           string eg, ['Sans' | 'Courier' | 'Helvetica' ...]
    position                   (x,y)
    rotation                   [ angle in degrees 'vertical' | 'horizontal'
    size or fontsize           [ size in points | relative size eg 'smaller', 'x-large' ]
    style or fontstyle         [ 'normal' | 'italic' | 'oblique']
    text                       string
    transform                  a matplotlib.transform transformation instance
    variant                    [ 'normal' | 'small-caps' ]
    verticalalignment or va    [ 'center' | 'top' | 'bottom' | 'baseline' ]
    visible                    [True | False]
    weight or fontweight       [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
    x                          float
    y                          float
    zorder                     any number
    ========================== =========================================================================
    """




# TODO : This function may move into the Text class as a method. As a
# matter of fact, The information from the _get_textbox function
# should be available during the Text._get_layout() call, which is
# called within the _get_textbox. So, it would better to move this
# function as a method with some refactoring of _get_layout method.

def _get_textbox(text, renderer):
    """
    Calculate the bounding box of the text. Unlike
    :meth:`matplotlib.text.Text.get_extents` method, The bbox size of
    the text before the rotation is calculated.
    """

    projected_xs = []
    projected_ys = []

    theta = text.get_rotation()/180.*math.pi
    tr = mtransforms.Affine2D().rotate(-theta)

    for t, wh, x, y in text._get_layout(renderer)[1]:
        w, h = wh


        xt1, yt1 = tr.transform_point((x, y))
        xt2, yt2 = xt1+w, yt1+h

        projected_xs.extend([xt1, xt2])
        projected_ys.extend([yt1, yt2])


    xt_box, yt_box = min(projected_xs), min(projected_ys)
    w_box, h_box = max(projected_xs) - xt_box, max(projected_ys) - yt_box

    tr = mtransforms.Affine2D().rotate(theta)

    x_box, y_box = tr.transform_point((xt_box, yt_box))

    return x_box, y_box, w_box, h_box



class Text(Artist):
    """
    Handle storing and drawing of text in window or data coordinates.
    """
    zorder = 3
    def __str__(self):
        return "Text(%g,%g,%s)"%(self._y,self._y,repr(self._text))

    def __init__(self,
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 linespacing=None,
                 **kwargs
                 ):
        """
        Create a :class:`~matplotlib.text.Text` instance at *x*, *y*
        with string *text*.

        Valid kwargs are
        %(Text)s
        """

        Artist.__init__(self)
        self.cached = maxdict(5)
        self._x, self._y = x, y

        if color is None: color = rcParams['text.color']
        if fontproperties is None: fontproperties=FontProperties()
        elif is_string_like(fontproperties): fontproperties=FontProperties(fontproperties)

        self.set_text(text)
        self.set_color(color)
        self._verticalalignment = verticalalignment
        self._horizontalalignment = horizontalalignment
        self._multialignment = multialignment
        self._rotation = rotation
        self._fontproperties = fontproperties
        self._bbox = None
        self._bbox_patch = None # a FancyBboxPatch instance
        self._renderer = None
        if linespacing is None:
            linespacing = 1.2   # Maybe use rcParam later.
        self._linespacing = linespacing
        self.update(kwargs)
        #self.set_bbox(dict(pad=0))

    def contains(self,mouseevent):
        """Test whether the mouse event occurred in the patch.

        In the case of text, a hit is true anywhere in the
        axis-aligned bounding-box containing the text.

        Returns True or False.
        """
        if callable(self._contains): return self._contains(self,mouseevent)

        if not self.get_visible() or self._renderer is None:
            return False,{}

        l,b,w,h = self.get_window_extent().bounds

        r = l+w
        t = b+h
        xyverts = (l,b), (l, t), (r, t), (r, b)
        x, y = mouseevent.x, mouseevent.y
        inside = nxutils.pnpoly(x, y, xyverts)

        return inside,{}

    def _get_xy_display(self):
        'get the (possibly unit converted) transformed x, y in display coords'
        x, y = self.get_position()
        return self.get_transform().transform_point((x,y))

    def _get_multialignment(self):
        if self._multialignment is not None: return self._multialignment
        else: return self._horizontalalignment

    def get_rotation(self):
        'return the text angle as float in degrees'
        return get_rotation(self._rotation)  # string_or_number -> number

    def update_from(self, other):
        'Copy properties from other to self'
        Artist.update_from(self, other)
        self._color = other._color
        self._multialignment = other._multialignment
        self._verticalalignment = other._verticalalignment
        self._horizontalalignment = other._horizontalalignment
        self._fontproperties = other._fontproperties.copy()
        self._rotation = other._rotation
        self._picker = other._picker
        self._linespacing = other._linespacing

    def _get_layout(self, renderer):
        key = self.get_prop_tup()
        if key in self.cached: return self.cached[key]

        horizLayout = []

        thisx, thisy  = 0.0, 0.0
        xmin, ymin    = 0.0, 0.0
        width, height = 0.0, 0.0
        lines = self._text.split('\n')

        whs = np.zeros((len(lines), 2))
        horizLayout = np.zeros((len(lines), 4))

        # Find full vertical extent of font,
        # including ascenders and descenders:
        tmp, heightt, bl = renderer.get_text_width_height_descent(
                'lp', self._fontproperties, ismath=False)
        offsety = heightt * self._linespacing

        baseline = None
        for i, line in enumerate(lines):
            clean_line, ismath = self.is_math_text(line)
            w, h, d = renderer.get_text_width_height_descent(
                clean_line, self._fontproperties, ismath=ismath)
            if baseline is None:
                baseline = h - d
            whs[i] = w, h
            horizLayout[i] = thisx, thisy, w, h
            thisy -= offsety
            width = max(width, w)

        ymin = horizLayout[-1][1]
        ymax = horizLayout[0][1] + horizLayout[0][3]
        height = ymax-ymin
        xmax = xmin + width

        # get the rotation matrix
        M = Affine2D().rotate_deg(self.get_rotation())

        offsetLayout = np.zeros((len(lines), 2))
        offsetLayout[:] = horizLayout[:, 0:2]
        # now offset the individual text lines within the box
        if len(lines)>1: # do the multiline aligment
            malign = self._get_multialignment()
            if malign == 'center':
                offsetLayout[:, 0] += width/2.0 - horizLayout[:, 2] / 2.0
            elif malign == 'right':
                offsetLayout[:, 0] += width - horizLayout[:, 2]

        # the corners of the unrotated bounding box
        cornersHoriz = np.array(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)],
            np.float_)
        # now rotate the bbox
        cornersRotated = M.transform(cornersHoriz)

        txs = cornersRotated[:, 0]
        tys = cornersRotated[:, 1]

        # compute the bounds of the rotated box
        xmin, xmax = txs.min(), txs.max()
        ymin, ymax = tys.min(), tys.max()
        width  = xmax - xmin
        height = ymax - ymin

        # Now move the box to the targe position offset the display bbox by alignment
        halign = self._horizontalalignment
        valign = self._verticalalignment

        # compute the text location in display coords and the offsets
        # necessary to align the bbox with that location
        if halign=='center':  offsetx = (xmin + width/2.0)
        elif halign=='right': offsetx = (xmin + width)
        else: offsetx = xmin

        if valign=='center': offsety = (ymin + height/2.0)
        elif valign=='top': offsety  = (ymin + height)
        elif valign=='baseline': offsety = (ymin + height) - baseline
        else: offsety = ymin

        xmin -= offsetx
        ymin -= offsety

        bbox = Bbox.from_bounds(xmin, ymin, width, height)

        # now rotate the positions around the first x,y position
        xys = M.transform(offsetLayout)
        xys -= (offsetx, offsety)

        xs, ys = xys[:, 0], xys[:, 1]

        ret = bbox, zip(lines, whs, xs, ys)
        self.cached[key] = ret
        return ret

    def set_bbox(self, rectprops):
        """
        Draw a bounding box around self.  rectprops are any settable
        properties for a rectangle, eg facecolor='red', alpha=0.5.

          t.set_bbox(dict(facecolor='red', alpha=0.5))

        If rectprops has "boxstyle" key. A FancyBboxPatch
        is initialized with rectprops and will be drawn. The mutation
        scale of the FancyBboxPath is set to the fontsize.

        ACCEPTS: rectangle prop dict
        """

        # The self._bbox_patch object is created only if rectprops has
        # boxstyle key. Otherwise, self._bbox will be set to the
        # rectprops and the bbox will be drawn using bbox_artist
        # function. This is to keep the backward compatibility.

        if rectprops is not None and "boxstyle" in rectprops:
            props = rectprops.copy()
            boxstyle = props.pop("boxstyle")
            bbox_transmuter = props.pop("bbox_transmuter", None)

            self._bbox_patch = FancyBboxPatch((0., 0.),
                                              1., 1.,
                                              boxstyle=boxstyle,
                                              bbox_transmuter=bbox_transmuter,
                                              transform=mtransforms.IdentityTransform(),
                                              **props)
            self._bbox = None
        else:
            self._bbox_patch = None
            self._bbox = rectprops


    def get_bbox_patch(self):
        """
        Return the bbox Patch object. Returns None if the the
        FancyBboxPatch is not made.
        """
        return self._bbox_patch


    def update_bbox_position_size(self, renderer):
        """
        Update the location and the size of the bbox. This method
        should be used when the position and size of the bbox needs to
        be updated before actually drawing the bbox.
        """

        # For arrow_patch, use textbox as patchA by default.

        if not isinstance(self.arrow_patch, FancyArrowPatch):
            return

        if self._bbox_patch:

            trans = self.get_transform()

            # don't use self.get_position here, which refers to text position
            # in Text, and dash position in TextWithDash:
            posx = float(self.convert_xunits(self._x))
            posy = float(self.convert_yunits(self._y))

            posx, posy = trans.transform_point((posx, posy))

            x_box, y_box, w_box, h_box = _get_textbox(self, renderer)
            self._bbox_patch.set_bounds(0., 0.,
                                        w_box, h_box)
            theta = self.get_rotation()/180.*math.pi
            tr = mtransforms.Affine2D().rotate(theta)
            tr = tr.translate(posx+x_box, posy+y_box)
            self._bbox_patch.set_transform(tr)
            fontsize_in_pixel = renderer.points_to_pixels(self.get_size())
            self._bbox_patch.set_mutation_scale(fontsize_in_pixel)
            #self._bbox_patch.draw(renderer)

        else:
            props = self._bbox
            if props is None: props = {}
            props = props.copy() # don't want to alter the pad externally
            pad = props.pop('pad', 4)
            pad = renderer.points_to_pixels(pad)
            bbox = self.get_window_extent(renderer)
            l,b,w,h = bbox.bounds
            l-=pad/2.
            b-=pad/2.
            w+=pad
            h+=pad
            r = Rectangle(xy=(l,b),
                          width=w,
                          height=h,
                          )
            r.set_transform(mtransforms.IdentityTransform())
            r.set_clip_on( False )
            r.update(props)

            self.arrow_patch.set_patchA(r)

    def _draw_bbox(self, renderer, posx, posy):

        """ Update the location and the size of the bbox
        (FancyBoxPatch), and draw
        """

        x_box, y_box, w_box, h_box = _get_textbox(self, renderer)
        self._bbox_patch.set_bounds(0., 0.,
                                    w_box, h_box)
        theta = self.get_rotation()/180.*math.pi
        tr = mtransforms.Affine2D().rotate(theta)
        tr = tr.translate(posx+x_box, posy+y_box)
        self._bbox_patch.set_transform(tr)
        fontsize_in_pixel = renderer.points_to_pixels(self.get_size())
        self._bbox_patch.set_mutation_scale(fontsize_in_pixel)
        self._bbox_patch.draw(renderer)


    def draw(self, renderer):
        """
        Draws the :class:`Text` object to the given *renderer*.
        """
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible(): return
        if self._text=='': return

        bbox, info = self._get_layout(renderer)
        trans = self.get_transform()


        # don't use self.get_position here, which refers to text position
        # in Text, and dash position in TextWithDash:
        posx = float(self.convert_xunits(self._x))
        posy = float(self.convert_yunits(self._y))

        posx, posy = trans.transform_point((posx, posy))
        canvasw, canvash = renderer.get_canvas_width_height()

        # draw the FancyBboxPatch
        if self._bbox_patch:
            self._draw_bbox(renderer, posx, posy)

        gc = renderer.new_gc()
        gc.set_foreground(self._color)
        gc.set_alpha(self._alpha)
        gc.set_url(self._url)
        if self.get_clip_on():
            gc.set_clip_rectangle(self.clipbox)

        if self._bbox:
            bbox_artist(self, renderer, self._bbox)
        angle = self.get_rotation()



        if rcParams['text.usetex']:
            for line, wh, x, y in info:
                x = x + posx
                y = y + posy
                if renderer.flipy():
                    y = canvash-y
                clean_line, ismath = self.is_math_text(line)

                renderer.draw_tex(gc, x, y, clean_line,
                                  self._fontproperties, angle)
            return

        for line, wh, x, y in info:
            x = x + posx
            y = y + posy
            if renderer.flipy():
                y = canvash-y
            clean_line, ismath = self.is_math_text(line)

            renderer.draw_text(gc, x, y, clean_line,
                               self._fontproperties, angle,
                               ismath=ismath)

    def get_color(self):
        "Return the color of the text"
        return self._color

    def get_fontproperties(self):
        "Return the :class:`~font_manager.FontProperties` object"
        return self._fontproperties

    def get_font_properties(self):
        'alias for get_fontproperties'
        return self.get_fontproperties

    def get_family(self):
        "Return the list of font families used for font lookup"
        return self._fontproperties.get_family()

    def get_fontfamily(self):
        'alias for get_family'
        return self.get_family()

    def get_name(self):
        "Return the font name as string"
        return self._fontproperties.get_name()

    def get_style(self):
        "Return the font style as string"
        return self._fontproperties.get_style()

    def get_size(self):
        "Return the font size as integer"
        return self._fontproperties.get_size_in_points()

    def get_variant(self):
        "Return the font variant as a string"
        return self._fontproperties.get_variant()

    def get_fontvariant(self):
        'alias for get_variant'
        return self.get_variant()

    def get_weight(self):
        "Get the font weight as string or number"
        return self._fontproperties.get_weight()

    def get_fontname(self):
        'alias for get_name'
        return self.get_name()

    def get_fontstyle(self):
        'alias for get_style'
        return self.get_style()

    def get_fontsize(self):
        'alias for get_size'
        return self.get_size()

    def get_fontweight(self):
        'alias for get_weight'
        return self.get_weight()

    def get_stretch(self):
        'Get the font stretch as a string or number'
        return self._fontproperties.get_stretch()

    def get_fontstretch(self):
        'alias for get_stretch'
        return self.get_stretch()

    def get_ha(self):
        'alias for get_horizontalalignment'
        return self.get_horizontalalignment()

    def get_horizontalalignment(self):
        """
        Return the horizontal alignment as string.  Will be one of
        'left', 'center' or 'right'.
        """
        return self._horizontalalignment


    def get_position(self):
        "Return the position of the text as a tuple (*x*, *y*)"
        x = float(self.convert_xunits(self._x))
        y = float(self.convert_yunits(self._y))
        return x, y

    def get_prop_tup(self):
        """
        Return a hashable tuple of properties.

        Not intended to be human readable, but useful for backends who
        want to cache derived information about text (eg layouts) and
        need to know if the text has changed.
        """
        x, y = self.get_position()
        return (x, y, self._text, self._color,
                self._verticalalignment, self._horizontalalignment,
                hash(self._fontproperties), self._rotation,
                self.figure.dpi, id(self._renderer),
                )

    def get_text(self):
        "Get the text as string"
        return self._text

    def get_va(self):
        'alias for :meth:`getverticalalignment`'
        return self.get_verticalalignment()

    def get_verticalalignment(self):
        """
        Return the vertical alignment as string.  Will be one of
        'top', 'center', 'bottom' or 'baseline'.
        """
        return self._verticalalignment

    def get_window_extent(self, renderer=None, dpi=None):
        '''
        Return a :class:`~matplotlib.transforms.Bbox` object bounding
        the text, in display units.

        In addition to being used internally, this is useful for
        specifying clickable regions in a png file on a web page.

        *renderer* defaults to the _renderer attribute of the text
        object.  This is not assigned until the first execution of
        :meth:`draw`, so you must use this kwarg if you want
        to call :meth:`get_window_extent` prior to the first
        :meth:`draw`.  For getting web page regions, it is
        simpler to call the method after saving the figure.

        *dpi* defaults to self.figure.dpi; the renderer dpi is
        irrelevant.  For the web application, if figure.dpi is not
        the value used when saving the figure, then the value that
        was used must be specified as the *dpi* argument.
        '''
        #return _unit_box
        if not self.get_visible(): return Bbox.unit()
        if dpi is not None:
            dpi_orig = self.figure.dpi
            self.figure.dpi = dpi
        if self._text == '':
            tx, ty = self._get_xy_display()
            return Bbox.from_bounds(tx,ty,0,0)

        if renderer is not None:
            self._renderer = renderer
        if self._renderer is None:
            raise RuntimeError('Cannot get window extent w/o renderer')

        bbox, info = self._get_layout(self._renderer)
        x, y = self.get_position()
        x, y = self.get_transform().transform_point((x, y))
        bbox = bbox.translated(x, y)
        if dpi is not None:
            self.figure.dpi = dpi_orig
        return bbox

    def set_backgroundcolor(self, color):
        """
        Set the background color of the text by updating the bbox.

        .. seealso::
            :meth:`set_bbox`

        ACCEPTS: any matplotlib color
        """
        if self._bbox is None:
            self._bbox = dict(facecolor=color, edgecolor=color)
        else:
            self._bbox.update(dict(facecolor=color))



    def set_color(self, color):
        """
        Set the foreground color of the text

        ACCEPTS: any matplotlib color
        """
        # Make sure it is hashable, or get_prop_tup will fail.
        try:
            hash(color)
        except TypeError:
            color = tuple(color)
        self._color = color

    def set_ha(self, align):
        'alias for set_horizontalalignment'
        self.set_horizontalalignment(align)

    def set_horizontalalignment(self, align):
        """
        Set the horizontal alignment to one of

        ACCEPTS: [ 'center' | 'right' | 'left' ]
        """
        legal = ('center', 'right', 'left')
        if align not in legal:
            raise ValueError('Horizontal alignment must be one of %s' % str(legal))
        self._horizontalalignment = align

    def set_ma(self, align):
        'alias for set_verticalalignment'
        self.set_multialignment(align)


    def set_multialignment(self, align):
        """
        Set the alignment for multiple lines layout.  The layout of the
        bounding box of all the lines is determined bu the horizontalalignment
        and verticalalignment properties, but the multiline text within that
        box can be

        ACCEPTS: ['left' | 'right' | 'center' ]
        """
        legal = ('center', 'right', 'left')
        if align not in legal:
            raise ValueError('Horizontal alignment must be one of %s' % str(legal))
        self._multialignment = align

    def set_linespacing(self, spacing):
        """
        Set the line spacing as a multiple of the font size.
        Default is 1.2.

        ACCEPTS: float (multiple of font size)
        """
        self._linespacing = spacing

    def set_family(self, fontname):
        """
        Set the font family.  May be either a single string, or a list
        of strings in decreasing priority.  Each string may be either
        a real font name or a generic font class name.  If the latter,
        the specific font names will be looked up in the
        :file:`matplotlibrc` file.

        ACCEPTS: [ FONTNAME | 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
        """
        self._fontproperties.set_family(fontname)

    def set_variant(self, variant):
        """
        Set the font variant, either 'normal' or 'small-caps'.

        ACCEPTS: [ 'normal' | 'small-caps' ]
        """
        self._fontproperties.set_variant(variant)

    def set_fontvariant(self, variant):
        'alias for set_variant'
        return self.set_variant(variant)

    def set_name(self, fontname):
        """alias for set_family"""
        return self.set_family(fontname)

    def set_fontname(self, fontname):
        """alias for set_family"""
        self.set_family(fontname)

    def set_style(self, fontstyle):
        """
        Set the font style.

        ACCEPTS: [ 'normal' | 'italic' | 'oblique']
        """
        self._fontproperties.set_style(fontstyle)

    def set_fontstyle(self, fontstyle):
        'alias for set_style'
        return self.set_style(fontstyle)

    def set_size(self, fontsize):
        """
        Set the font size.  May be either a size string, relative to
        the default font size, or an absolute font size in points.

        ACCEPTS: [ size in points | 'xx-small' | 'x-small' | 'small' | 'medium' | 'large' | 'x-large' | 'xx-large' ]
        """
        self._fontproperties.set_size(fontsize)

    def set_fontsize(self, fontsize):
        'alias for set_size'
        return self.set_size(fontsize)

    def set_weight(self, weight):
        """
        Set the font weight.

        ACCEPTS: [ a numeric value in range 0-1000 | 'ultralight' | 'light' | 'normal' | 'regular' | 'book' | 'medium' | 'roman' | 'semibold' | 'demibold' | 'demi' | 'bold' | 'heavy' | 'extra bold' | 'black' ]
        """
        self._fontproperties.set_weight(weight)

    def set_fontweight(self, weight):
        'alias for set_weight'
        return self.set_weight(weight)

    def set_stretch(self, stretch):
        """
        Set the font stretch (horizontal condensation or expansion).

        ACCEPTS: [ a numeric value in range 0-1000 | 'ultra-condensed' | 'extra-condensed' | 'condensed' | 'semi-condensed' | 'normal' | 'semi-expanded' | 'expanded' | 'extra-expanded' | 'ultra-expanded' ]
        """
        self._fontproperties.set_stretch(stretch)

    def set_fontstretch(self, stretch):
        'alias for set_stretch'
        return self.set_stretch(stretch)

    def set_position(self, xy):
        """
        Set the (*x*, *y*) position of the text

        ACCEPTS: (x,y)
        """
        self.set_x(xy[0])
        self.set_y(xy[1])

    def set_x(self, x):
        """
        Set the *x* position of the text

        ACCEPTS: float
        """
        self._x = x


    def set_y(self, y):
        """
        Set the *y* position of the text

        ACCEPTS: float
        """
        self._y = y


    def set_rotation(self, s):
        """
        Set the rotation of the text

        ACCEPTS: [ angle in degrees | 'vertical' | 'horizontal' ]
        """
        self._rotation = s



    def set_va(self, align):
        'alias for set_verticalalignment'
        self.set_verticalalignment(align)

    def set_verticalalignment(self, align):
        """
        Set the vertical alignment

        ACCEPTS: [ 'center' | 'top' | 'bottom' | 'baseline' ]
        """
        legal = ('top', 'bottom', 'center', 'baseline')
        if align not in legal:
            raise ValueError('Vertical alignment must be one of %s' % str(legal))

        self._verticalalignment = align

    def set_text(self, s):
        """
        Set the text string *s*

        It may contain newlines (``\\n``) or math in LaTeX syntax.

        ACCEPTS: string or anything printable with '%s' conversion.
        """
        self._text = '%s' % (s,)

    def is_math_text(self, s):
        """
        Returns True if the given string *s* contains any mathtext.
        """
        # Did we find an even number of non-escaped dollar signs?
        # If so, treat is as math text.
        dollar_count = s.count(r'$') - s.count(r'\$')
        even_dollars = (dollar_count > 0 and dollar_count % 2 == 0)

        if rcParams['text.usetex']:
            return s, 'TeX'

        if even_dollars:
            return s, True
        else:
            return s.replace(r'\$', '$'), False

    def set_fontproperties(self, fp):
        """
        Set the font properties that control the text.  *fp* must be a
        :class:`matplotlib.font_manager.FontProperties` object.

        ACCEPTS: a :class:`matplotlib.font_manager.FontProperties` instance
        """
        if is_string_like(fp):
            fp = FontProperties(fp)
        self._fontproperties = fp.copy()

    def set_font_properties(self, fp):
        'alias for set_fontproperties'
        self.set_fontproperties(fp)

artist.kwdocd['Text'] = artist.kwdoc(Text)
Text.__init__.im_func.__doc__ = cbook.dedent(Text.__init__.__doc__) % artist.kwdocd


class TextWithDash(Text):
    """
    This is basically a :class:`~matplotlib.text.Text` with a dash
    (drawn with a :class:`~matplotlib.lines.Line2D`) before/after
    it. It is intended to be a drop-in replacement for
    :class:`~matplotlib.text.Text`, and should behave identically to
    it when *dashlength* = 0.0.

    The dash always comes between the point specified by
    :meth:`~matplotlib.text.Text.set_position` and the text. When a
    dash exists, the text alignment arguments (*horizontalalignment*,
    *verticalalignment*) are ignored.

    *dashlength* is the length of the dash in canvas units.
    (default = 0.0).

    *dashdirection* is one of 0 or 1, where 0 draws the dash after the
    text and 1 before.  (default = 0).

    *dashrotation* specifies the rotation of the dash, and should
    generally stay *None*. In this case
    :meth:`~matplotlib.text.TextWithDash.get_dashrotation` returns
    :meth:`~matplotlib.text.Text.get_rotation`.  (I.e., the dash takes
    its rotation from the text's rotation). Because the text center is
    projected onto the dash, major deviations in the rotation cause
    what may be considered visually unappealing results.
    (default = *None*)

    *dashpad* is a padding length to add (or subtract) space
    between the text and the dash, in canvas units.
    (default = 3)

    *dashpush* "pushes" the dash and text away from the point
    specified by :meth:`~matplotlib.text.Text.set_position` by the
    amount in canvas units.  (default = 0)

    .. note::
        The alignment of the two objects is based on the bounding box
        of the :class:`~matplotlib.text.Text`, as obtained by
        :meth:`~matplotlib.artist.Artist.get_window_extent`.  This, in
        turn, appears to depend on the font metrics as given by the
        rendering backend. Hence the quality of the "centering" of the
        label text with respect to the dash varies depending on the
        backend used.

    .. note::
        I'm not sure that I got the
        :meth:`~matplotlib.text.TextWithDash.get_window_extent` right,
        or whether that's sufficient for providing the object bounding
        box.
    """
    __name__ = 'textwithdash'

    def __str__(self):
        return "TextWithDash(%g,%g,%s)"%(self._x,self._y,repr(self._text))
    def __init__(self,
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='center',
                 horizontalalignment='center',
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 linespacing=None,
                 dashlength=0.0,
                 dashdirection=0,
                 dashrotation=None,
                 dashpad=3,
                 dashpush=0,
                 ):

        Text.__init__(self, x=x, y=y, text=text, color=color,
                      verticalalignment=verticalalignment,
                      horizontalalignment=horizontalalignment,
                      multialignment=multialignment,
                      fontproperties=fontproperties,
                      rotation=rotation,
                      linespacing=linespacing)

        # The position (x,y) values for text and dashline
        # are bogus as given in the instantiation; they will
        # be set correctly by update_coords() in draw()

        self.dashline = Line2D(xdata=(x, x),
                               ydata=(y, y),
                               color='k',
                               linestyle='-')

        self._dashx = float(x)
        self._dashy = float(y)
        self._dashlength = dashlength
        self._dashdirection = dashdirection
        self._dashrotation = dashrotation
        self._dashpad = dashpad
        self._dashpush = dashpush

        #self.set_bbox(dict(pad=0))

    def get_position(self):
        "Return the position of the text as a tuple (*x*, *y*)"
        x = float(self.convert_xunits(self._dashx))
        y = float(self.convert_yunits(self._dashy))
        return x, y

    def get_prop_tup(self):
        """
        Return a hashable tuple of properties.

        Not intended to be human readable, but useful for backends who
        want to cache derived information about text (eg layouts) and
        need to know if the text has changed.
        """
        props = [p for p in Text.get_prop_tup(self)]
        props.extend([self._x, self._y, self._dashlength, self._dashdirection, self._dashrotation, self._dashpad, self._dashpush])
        return tuple(props)

    def draw(self, renderer):
        """
        Draw the :class:`TextWithDash` object to the given *renderer*.
        """
        self.update_coords(renderer)
        Text.draw(self, renderer)
        if self.get_dashlength() > 0.0:
            self.dashline.draw(renderer)

    def update_coords(self, renderer):
        """
        Computes the actual *x*, *y* coordinates for text based on the
        input *x*, *y* and the *dashlength*. Since the rotation is
        with respect to the actual canvas's coordinates we need to map
        back and forth.
        """
        dashx, dashy = self.get_position()
        dashlength = self.get_dashlength()
        # Shortcircuit this process if we don't have a dash
        if dashlength == 0.0:
            self._x, self._y = dashx, dashy
            return

        dashrotation = self.get_dashrotation()
        dashdirection = self.get_dashdirection()
        dashpad = self.get_dashpad()
        dashpush = self.get_dashpush()

        angle = get_rotation(dashrotation)
        theta = np.pi*(angle/180.0+dashdirection-1)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        transform = self.get_transform()

        # Compute the dash end points
        # The 'c' prefix is for canvas coordinates
        cxy = transform.transform_point((dashx, dashy))
        cd = np.array([cos_theta, sin_theta])
        c1 = cxy+dashpush*cd
        c2 = cxy+(dashpush+dashlength)*cd

        inverse = transform.inverted()
        (x1, y1) = inverse.transform_point(tuple(c1))
        (x2, y2) = inverse.transform_point(tuple(c2))
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
        we = Text.get_window_extent(self, renderer=renderer)
        w, h = we.width, we.height
        # Watch for zeros
        if sin_theta == 0.0:
            dx = w
            dy = 0.0
        elif cos_theta == 0.0:
            dx = 0.0
            dy = h
        else:
            tan_theta = sin_theta/cos_theta
            dx = w
            dy = w*tan_theta
            if dy > h or dy < -h:
                dy = h
                dx = h/tan_theta
        cwd = np.array([dx, dy])/2
        cwd *= 1+dashpad/np.sqrt(np.dot(cwd,cwd))
        cw = c2+(dashdirection*2-1)*cwd

        newx, newy = inverse.transform_point(tuple(cw))
        self._x, self._y = newx, newy

        # Now set the window extent
        # I'm not at all sure this is the right way to do this.
        we = Text.get_window_extent(self, renderer=renderer)
        self._twd_window_extent = we.frozen()
        self._twd_window_extent.update_from_data_xy(np.array([c1]), False)

        # Finally, make text align center
        Text.set_horizontalalignment(self, 'center')
        Text.set_verticalalignment(self, 'center')

    def get_window_extent(self, renderer=None):
        '''
        Return a :class:`~matplotlib.transforms.Bbox` object bounding
        the text, in display units.

        In addition to being used internally, this is useful for
        specifying clickable regions in a png file on a web page.

        *renderer* defaults to the _renderer attribute of the text
        object.  This is not assigned until the first execution of
        :meth:`draw`, so you must use this kwarg if you want
        to call :meth:`get_window_extent` prior to the first
        :meth:`draw`.  For getting web page regions, it is
        simpler to call the method after saving the figure.
        '''
        self.update_coords(renderer)
        if self.get_dashlength() == 0.0:
            return Text.get_window_extent(self, renderer=renderer)
        else:
            return self._twd_window_extent

    def get_dashlength(self):
        """
        Get the length of the dash.
        """
        return self._dashlength

    def set_dashlength(self, dl):
        """
        Set the length of the dash.

        ACCEPTS: float (canvas units)
        """
        self._dashlength = dl

    def get_dashdirection(self):
        """
        Get the direction dash.  1 is before the text and 0 is after.
        """
        return self._dashdirection

    def set_dashdirection(self, dd):
        """
        Set the direction of the dash following the text.
        1 is before the text and 0 is after. The default
        is 0, which is what you'd want for the typical
        case of ticks below and on the left of the figure.

        ACCEPTS: int (1 is before, 0 is after)
        """
        self._dashdirection = dd

    def get_dashrotation(self):
        """
        Get the rotation of the dash in degrees.
        """
        if self._dashrotation == None:
            return self.get_rotation()
        else:
            return self._dashrotation

    def set_dashrotation(self, dr):
        """
        Set the rotation of the dash, in degrees

        ACCEPTS: float (degrees)
        """
        self._dashrotation = dr

    def get_dashpad(self):
        """
        Get the extra spacing between the dash and the text, in canvas units.
        """
        return self._dashpad

    def set_dashpad(self, dp):
        """
        Set the "pad" of the TextWithDash, which is the extra spacing
        between the dash and the text, in canvas units.

        ACCEPTS: float (canvas units)
        """
        self._dashpad = dp

    def get_dashpush(self):
        """
        Get the extra spacing between the dash and the specified text
        position, in canvas units.
        """
        return self._dashpush

    def set_dashpush(self, dp):
        """
        Set the "push" of the TextWithDash, which
        is the extra spacing between the beginning
        of the dash and the specified position.

        ACCEPTS: float (canvas units)
        """
        self._dashpush = dp


    def set_position(self, xy):
        """
        Set the (*x*, *y*) position of the :class:`TextWithDash`.

        ACCEPTS: (x, y)
        """
        self.set_x(xy[0])
        self.set_y(xy[1])

    def set_x(self, x):
        """
        Set the *x* position of the :class:`TextWithDash`.

        ACCEPTS: float
        """
        self._dashx = float(x)

    def set_y(self, y):
        """
        Set the *y* position of the :class:`TextWithDash`.

        ACCEPTS: float
        """
        self._dashy = float(y)

    def set_transform(self, t):
        """
        Set the :class:`matplotlib.transforms.Transform` instance used
        by this artist.

        ACCEPTS: a :class:`matplotlib.transforms.Transform` instance
        """
        Text.set_transform(self, t)
        self.dashline.set_transform(t)

    def get_figure(self):
        'return the figure instance the artist belongs to'
        return self.figure

    def set_figure(self, fig):
        """
        Set the figure instance the artist belong to.

        ACCEPTS: a :class:`matplotlib.figure.Figure` instance
        """
        Text.set_figure(self, fig)
        self.dashline.set_figure(fig)

artist.kwdocd['TextWithDash'] = artist.kwdoc(TextWithDash)

class Annotation(Text):
    """
    A :class:`~matplotlib.text.Text` class to make annotating things
    in the figure, such as :class:`~matplotlib.figure.Figure`,
    :class:`~matplotlib.axes.Axes`,
    :class:`~matplotlib.patches.Rectangle`, etc., easier.
    """
    def __str__(self):
        return "Annotation(%g,%g,%s)"%(self.xy[0],self.xy[1],repr(self._text))
    def __init__(self, s, xy,
                 xytext=None,
                 xycoords='data',
                 textcoords=None,
                 arrowprops=None,
                 **kwargs):
        """
        Annotate the *x*, *y* point *xy* with text *s* at *x*, *y*
        location *xytext*.  (If *xytext* = *None*, defaults to *xy*,
        and if *textcoords* = *None*, defaults to *xycoords*).

        *arrowprops*, if not *None*, is a dictionary of line properties
        (see :class:`matplotlib.lines.Line2D`) for the arrow that connects
        annotation to the point.

        If the dictionary has a key *arrowstyle*, a FancyArrowPatch
        instance is created with the given dictionary and is
        drawn. Otherwise, a YAArow patch instance is created and
        drawn. Valid keys for YAArow are


        =========   =============================================================
        Key         Description
        =========   =============================================================
        width       the width of the arrow in points
        frac        the fraction of the arrow length occupied by the head
        headwidth   the width of the base of the arrow head in points
        shrink      oftentimes it is convenient to have the arrowtip
                    and base a bit away from the text and point being
                    annotated.  If *d* is the distance between the text and
                    annotated point, shrink will shorten the arrow so the tip
                    and base are shink percent of the distance *d* away from the
                    endpoints.  ie, ``shrink=0.05 is 5%%``
        ?           any key for :class:`matplotlib.patches.polygon`
        =========   =============================================================


        Valid keys for FancyArrowPatch are


        ===============  ======================================================
        Key              Description
        ===============  ======================================================
        arrowstyle       the arrow style
        connectionstyle  the connection style
        relpos           default is (0.5, 0.5)
        patchA           default is bounding box of the text
        patchB           default is None
        shrinkA          default is 2 points
        shrinkB          default is 2 points
        mutation_scale   default is text size (in points)
        mutation_aspect  default is 1.
        ?                any key for :class:`matplotlib.patches.PathPatch`
        ===============  ======================================================


        *xycoords* and *textcoords* are strings that indicate the
        coordinates of *xy* and *xytext*.

        =================   ===================================================
        Property            Description
        =================   ===================================================
        'figure points'     points from the lower left corner of the figure
        'figure pixels'     pixels from the lower left corner of the figure
        'figure fraction'   0,0 is lower left of figure and 1,1 is upper, right
        'axes points'       points from lower left corner of axes
        'axes pixels'       pixels from lower left corner of axes
        'axes fraction'     0,1 is lower left of axes and 1,1 is upper right
        'data'              use the coordinate system of the object being
                            annotated (default)
        'offset points'     Specify an offset (in points) from the *xy* value

        'polar'             you can specify *theta*, *r* for the annotation,
                            even in cartesian plots.  Note that if you
                            are using a polar axes, you do not need
                            to specify polar for the coordinate
                            system since that is the native "data" coordinate
                            system.
        =================   ===================================================

        If a 'points' or 'pixels' option is specified, values will be
        added to the bottom-left and if negative, values will be
        subtracted from the top-right.  Eg::

          # 10 points to the right of the left border of the axes and
          # 5 points below the top border
          xy=(10,-5), xycoords='axes points'

        Additional kwargs are Text properties:

        %(Text)s

        """
        if xytext is None:
            xytext = xy
        if textcoords is None:
            textcoords = xycoords
        # we'll draw ourself after the artist we annotate by default
        x,y = self.xytext = xytext
        Text.__init__(self, x, y, s, **kwargs)
        self.xy = xy
        self.xycoords = xycoords
        self.textcoords = textcoords

        self.arrowprops = arrowprops

        self.arrow = None

        if arrowprops and arrowprops.has_key("arrowstyle"):

            self._arrow_relpos = arrowprops.pop("relpos", (0.5, 0.5))
            self.arrow_patch = FancyArrowPatch((0, 0), (1,1),
                                               **arrowprops)
        else:
            self.arrow_patch = None


    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def contains(self,event):
        t,tinfo = Text.contains(self,event)
        if self.arrow is not None:
            a,ainfo=self.arrow.contains(event)
            t = t or a

        # self.arrow_patch is currently not checked as this can be a line - JJ

        return t,tinfo


    def set_figure(self, fig):

        if self.arrow is not None:
            self.arrow.set_figure(fig)
        if self.arrow_patch is not None:
            self.arrow_patch.set_figure(fig)
        Artist.set_figure(self, fig)

    def _get_xy(self, x, y, s):
        if s=='data':
            trans = self.axes.transData
            x = float(self.convert_xunits(x))
            y = float(self.convert_yunits(y))
            return trans.transform_point((x, y))
        elif s=='offset points':
            # convert the data point
            dx, dy = self.xy

            # prevent recursion
            if self.xycoords == 'offset points':
                return self._get_xy(dx, dy, 'data')

            dx, dy = self._get_xy(dx, dy, self.xycoords)

            # convert the offset
            dpi = self.figure.get_dpi()
            x *= dpi/72.
            y *= dpi/72.

            # add the offset to the data point
            x += dx
            y += dy

            return x, y
        elif s=='polar':
            theta, r = x, y
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            trans = self.axes.transData
            return trans.transform_point((x,y))
        elif s=='figure points':
            #points from the lower left corner of the figure
            dpi = self.figure.dpi
            l,b,w,h = self.figure.bbox.bounds
            r = l+w
            t = b+h

            x *= dpi/72.
            y *= dpi/72.
            if x<0:
                x = r + x
            if y<0:
                y = t + y
            return x,y
        elif s=='figure pixels':
            #pixels from the lower left corner of the figure
            l,b,w,h = self.figure.bbox.bounds
            r = l+w
            t = b+h
            if x<0:
                x = r + x
            if y<0:
                y = t + y
            return x, y
        elif s=='figure fraction':
            #(0,0) is lower left, (1,1) is upper right of figure
            trans = self.figure.transFigure
            return trans.transform_point((x,y))
        elif s=='axes points':
            #points from the lower left corner of the axes
            dpi = self.figure.dpi
            l,b,w,h = self.axes.bbox.bounds
            r = l+w
            t = b+h
            if x<0:
                x = r + x*dpi/72.
            else:
                x = l + x*dpi/72.
            if y<0:
                y = t + y*dpi/72.
            else:
                y = b + y*dpi/72.
            return x, y
        elif s=='axes pixels':
            #pixels from the lower left corner of the axes

            l,b,w,h = self.axes.bbox.bounds
            r = l+w
            t = b+h
            if x<0:
                x = r + x
            else:
                x = l + x
            if y<0:
                y = t + y
            else:
                y = b + y
            return x, y
        elif s=='axes fraction':
            #(0,0) is lower left, (1,1) is upper right of axes
            trans = self.axes.transAxes
            return trans.transform_point((x, y))


    def update_positions(self, renderer):

        x, y = self.xytext
        self._x, self._y = self._get_xy(x, y, self.textcoords)


        x, y = self.xy
        x, y = self._get_xy(x, y, self.xycoords)

        ox0, oy0 = self._x, self._y
        ox1, oy1 = x, y

        if self.arrowprops:
            x0, y0 = x, y
            l,b,w,h = self.get_window_extent(renderer).bounds
            r = l+w
            t = b+h
            xc = 0.5*(l+r)
            yc = 0.5*(b+t)

            d = self.arrowprops.copy()

            # Use FancyArrowPatch if self.arrowprops has "arrowstyle" key.
            # Otherwise, fallback to YAArrow.

            #if d.has_key("arrowstyle"):
            if self.arrow_patch:

                # adjust the starting point of the arrow relative to
                # the textbox.
                # TODO : Rotation needs to be accounted.
                relpos = self._arrow_relpos
                bbox = self.get_window_extent(renderer)
                ox0 = bbox.x0 + bbox.width * relpos[0]
                oy0 = bbox.y0 + bbox.height * relpos[1]

                # The arrow will be drawn from (ox0, oy0) to (ox1,
                # oy1). It will be first clipped by patchA and patchB.
                # Then it will be shrinked by shirnkA and shrinkB
                # (in points). If patch A is not set, self.bbox_patch
                # is used.

                self.arrow_patch.set_positions((ox0, oy0), (ox1,oy1))
                mutation_scale = d.pop("mutation_scale", self.get_size())
                mutation_scale = renderer.points_to_pixels(mutation_scale)
                self.arrow_patch.set_mutation_scale(mutation_scale)

                if self._bbox_patch:
                    patchA = d.pop("patchA", self._bbox_patch)
                    self.arrow_patch.set_patchA(patchA)
                else:
                    patchA = d.pop("patchA", self._bbox)
                    self.arrow_patch.set_patchA(patchA)

            else:

                # pick the x,y corner of the text bbox closest to point
                # annotated
                dsu = [(abs(val-x0), val) for val in l, r, xc]
                dsu.sort()
                _, x = dsu[0]

                dsu = [(abs(val-y0), val) for val in b, t, yc]
                dsu.sort()
                _, y = dsu[0]

                shrink = d.pop('shrink', 0.0)

                theta = math.atan2(y-y0, x-x0)
                r = math.sqrt((y-y0)**2. + (x-x0)**2.)
                dx = shrink*r*math.cos(theta)
                dy = shrink*r*math.sin(theta)

                width = d.pop('width', 4)
                headwidth = d.pop('headwidth', 12)
                frac = d.pop('frac', 0.1)

                self.arrow = YAArrow(self.figure, (x0+dx,y0+dy), (x-dx, y-dy),
                                     width=width, headwidth=headwidth, frac=frac,
                                     **d)

                self.arrow.set_clip_box(self.get_clip_box())

    def draw(self, renderer):
        """
        Draw the :class:`Annotation` object to the given *renderer*.
        """
        self.update_positions(renderer)
        self.update_bbox_position_size(renderer)

        if self.arrow is not None:
            if self.arrow.figure is None and self.figure is not None:
                self.arrow.figure = self.figure
            self.arrow.draw(renderer)

        if self.arrow_patch is not None:
            if self.arrow_patch.figure is None and self.figure is not None:
                self.arrow_patch.figure = self.figure
            self.arrow_patch.draw(renderer)

        Text.draw(self, renderer)


artist.kwdocd['Annotation'] = Annotation.__init__.__doc__
