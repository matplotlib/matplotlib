"""
Figure and Axes text
"""
from __future__ import division
import re
from matplotlib import verbose
import matplotlib
import math
from artist import Artist
from cbook import enumerate, popd, is_string_like, maxdict, is_numlike
from font_manager import FontProperties
from matplotlib import rcParams
from patches import bbox_artist, YAArrow
from numerix import sin, cos, pi, cumsum, dot, asarray, array, \
     where, nonzero, equal, sqrt
from transforms import lbwh_to_bbox, bbox_all, identity_transform
from lines import Line2D

import matplotlib.nxutils as nxutils
import artist

def scanner(s):
    """
    Split a string into mathtext and non-mathtext parts.  mathtext is
    surrounded by $ symbols.  quoted \$ are ignored

    All slash quotes dollar signs are ignored

    The number of unquoted dollar signs must be even

    Return value is a list of (substring, inmath) tuples
    """
    if not len(s): return [(s, False)]
    #print 'testing', s, type(s)
    inddollar = nonzero(asarray(equal(s,'$')))
    quoted = dict([ (ind,1) for ind in nonzero(asarray(equal(s,'\\')))])
    indkeep = [ind for ind in inddollar if not quoted.has_key(ind-1)]
    if len(indkeep)==0:
        return [(s, False)]
    if len(indkeep)%2:
        raise ValueError('Illegal string "%s" (must have balanced dollar signs)'%s)

    Ns = len(s)

    indkeep = [ind for ind in indkeep]
    # make sure we start with the first element
    if indkeep[0]!=0: indkeep.insert(0,0)
    # and end with one past the end of the string
    indkeep.append(Ns+1)

    Nkeep = len(indkeep)
    results = []

    inmath = s[0] == '$'
    for i in range(Nkeep-1):
        i0, i1 = indkeep[i], indkeep[i+1]
        if not inmath:
            if i0>0: i0 +=1
        else:
            i1 += 1
        if i0>=Ns: break

        results.append((s[i0:i1], inmath))
        inmath = not inmath

    return results



def _process_text_args(override, fontdict=None, **kwargs):
    "Return an override dict.  See 'text' docstring for info"

    if fontdict is not None:
        override.update(fontdict)

    override.update(kwargs)
    return override

# Extracted from Text's method to serve as a function
def get_rotation(rotation):
    'return the text angle as float'
    if rotation in ('horizontal', None):
        angle = 0.
    elif rotation == 'vertical':
        angle = 90.
    else:
        angle = float(rotation)
    return angle%360

_unit_box = lbwh_to_bbox(0,0,1,1)


class Text(Artist):
    """
    Handle storing and drawing of text in window or data coordinates

    """
    # special case superscripting to speedup logplots
    _rgxsuper = re.compile('\$([\-+0-9]+)\^\{(-?[0-9]+)\}\$')

    zorder = 3
    def __init__(self,
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 **kwargs
                 ):
        """
        Create a Text instance at x,y with string text.  Valid kwargs are

            alpha: float
            animated: [True | False]
            backgroundcolor: any matplotlib color
            bbox: rectangle prop dict plus key 'pad' which is a pad in points
            clip_box: a matplotlib.transform.Bbox instance
            clip_on: [True | False]
            color: any matplotlib color
            family: [ 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
            figure: a matplotlib.figure.Figure instance
            fontproperties: a matplotlib.font_manager.FontProperties instance
            horizontalalignment or ha: [ 'center' | 'right' | 'left' ]
            label: any string
            lod: [True | False]
            multialignment: ['left' | 'right' | 'center' ]
            name or fontname: string eg, ['Sans' | 'Courier' | 'Helvetica' ...]
            position: (x,y)
            rotation: [ angle in degrees 'vertical' | 'horizontal'
            size or fontsize: [ size in points | relative size eg 'smaller', 'x-large' ]
            style or fontstyle: [ 'normal' | 'italic' | 'oblique']
            text: string
            transform: a matplotlib.transform transformation instance
            variant: [ 'normal' | 'small-caps' ]
            verticalalignment or va: [ 'center' | 'top' | 'bottom' ]
            visible: [True | False]
            weight or fontweight: [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
            x: float
            y: float
            zorder: any number
        """

        Artist.__init__(self)
        if not is_string_like(text):
            raise TypeError('text must be a string type')
        self.cached = maxdict(5)
        self._x, self._y = x, y

        if color is None: color = rcParams['text.color']
        if fontproperties is None: fontproperties=FontProperties()

        self.set_color(color)
        self.set_text(text)
        self._verticalalignment = verticalalignment
        self._horizontalalignment = horizontalalignment
        self._multialignment = multialignment
        self._rotation = rotation
        self._fontproperties = fontproperties
        self._bbox = None
        self._renderer = None
        self.update(kwargs)
        #self.set_bbox(dict(pad=0))


    def pick(self, mouseevent):
        """
        if the mouse click is inside the vertices defining the
        bounding box of the text, fire off a backend_bases.PickEvent
        """
        if not self.pickable(): return
        picker = self.get_picker()
        if callable(picker):
            hit, props = picker(self, mouseevent)
            if hit:
                self.figure.canvas.pick_event(mouseevent, self, **props)
        elif picker:
            l,b,w,h = self.get_window_extent().get_bounds()
            r = l+w
            t = b+h
            xyverts = (l,b), (l, t), (r, t), (r, b)
            x, y = mouseevent.x, mouseevent.y
            inside = nxutils.pnpoly(x, y, xyverts)
            if inside:
                self.figure.canvas.pick_event(mouseevent, self)

    def _get_multialignment(self):
        if self._multialignment is not None: return self._multialignment
        else: return self._horizontalalignment

    def get_rotation(self):
        'return the text angle as float'
        #return 0

#         if self._rotation in ('horizontal', None):
#             angle = 0.
#         elif self._rotation == 'vertical':
#             angle = 90.
#         else:
#             angle = float(self._rotation)
#         return angle%360

        # Since the get_rotation logic was extracted
        # into a function for TextWithDash, this
        # method could now read as follows.
        return get_rotation(self._rotation)

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

    def _get_layout(self, renderer):

        # layout the xylocs in display coords as if angle = zero and
        # then rotate them around self._x, self._y
        #return _unit_box
        key = self.get_prop_tup()
        if self.cached.has_key(key): return self.cached[key]
        horizLayout = []
        pad =2
        thisx, thisy = self.get_transform().xy_tup( (self._x, self._y) )
        width = 0
        height = 0

        xmin, ymin = thisx, thisy
        if self.is_math_text():
            lines = [self._text]
        else:
            lines = self._text.split('\n')

        whs = []
        tmp, heightt = renderer.get_text_width_height(
                'T', self._fontproperties, ismath=False)

        heightt += 3  # 3 pixel pad
        for line in lines:
            w,h = renderer.get_text_width_height(
                line, self._fontproperties, ismath=self.is_math_text())

            whs.append( (w,h) )
            offsety = heightt+pad
            horizLayout.append((line, thisx, thisy, w, h))
            thisy -= offsety  # now translate down by text height, window coords
            width = max(width, w)

        ymin = horizLayout[-1][2]
        ymax = horizLayout[0][2] + horizLayout[0][-1]
        height = ymax-ymin

        xmax = xmin + width
        # get the rotation matrix
        M = self.get_rotation_matrix(xmin, ymin)

        # the corners of the unrotated bounding box
        cornersHoriz = ( (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin) )
        offsetLayout = []
        # now offset the individual text lines within the box
        if len(lines)>1: # do the multiline aligment
            malign = self._get_multialignment()
            for line, thisx, thisy, w, h in horizLayout:
                if malign=='center': offsetx = width/2.0-w/2.0
                elif malign=='right': offsetx = width-w
                else: offsetx = 0
                thisx += offsetx
                offsetLayout.append( (thisx, thisy ))
        else: # no additional layout needed
            offsetLayout = [ (thisx, thisy) for line, thisx, thisy, w, h in horizLayout]

        # now rotate the bbox

        cornersRotated = [dot(M,array([[thisx],[thisy],[1]])) for thisx, thisy in cornersHoriz]

        txs = [float(v[0][0]) for v in cornersRotated]
        tys = [float(v[1][0]) for v in cornersRotated]

        # compute the bounds of the rotated box
        xmin, xmax = min(txs), max(txs)
        ymin, ymax = min(tys), max(tys)
        width  = xmax - xmin
        height = ymax - ymin

        # Now move the box to the targe position offset the display bbox by alignment
        halign = self._horizontalalignment
        valign = self._verticalalignment

        # compute the text location in display coords and the offsets
        # necessary to align the bbox with that location
        tx, ty = self.get_transform().xy_tup( (self._x, self._y) )

        if halign=='center':  offsetx = tx - (xmin + width/2.0)
        elif halign=='right': offsetx = tx - (xmin + width)
        else: offsetx = tx - xmin

        if valign=='center': offsety = ty - (ymin + height/2.0)
        elif valign=='top': offsety  = ty - (ymin + height)
        else: offsety = ty - ymin

        xmin += offsetx
        xmax += offsetx
        ymin += offsety
        ymax += offsety

        bbox = lbwh_to_bbox(xmin, ymin, width, height)


        # now rotate the positions around the first x,y position
        xys = [dot(M,array([[thisx],[thisy],[1]])) for thisx, thisy in offsetLayout]


        tx = [float(v[0][0])+offsetx for v in xys]
        ty = [float(v[1][0])+offsety for v in xys]

        # now inverse transform back to data coords
        xys = [self.get_transform().inverse_xy_tup( xy ) for xy in zip(tx, ty)]

        xs, ys = zip(*xys)

        ret = bbox, zip(lines, whs, xs, ys)
        self.cached[key] = ret
        return ret


    def set_bbox(self, rectprops):
        """
        Draw a bounding box around self.  rect props are any settable
        properties for a rectangle, eg facecolor='red', alpha=0.5.

          t.set_bbox(dict(facecolor='red', alpha=0.5))

        ACCEPTS: rectangle prop dict plus key 'pad' which is a pad in points
        """
        self._bbox = rectprops

    def draw(self, renderer):
        #return
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible(): return
        if self._text=='': return

        gc = renderer.new_gc()
        gc.set_foreground(self._color)
        gc.set_alpha(self._alpha)
        if self.get_clip_on():
            gc.set_clip_rectangle(self.clipbox.get_bounds())



        if self._bbox:
            bbox_artist(self, renderer, self._bbox)
        angle = self.get_rotation()

        ismath = self.is_math_text()

        if angle==0:
            #print 'text', self._text
            if ismath=='TeX': m = None
            else: m = self._rgxsuper.match(self._text)
            if m is not None:
                bbox, info = self._get_layout_super(self._renderer, m)
                base, xt, yt = info[0]
                renderer.draw_text(gc, xt, yt, base,
                                   self._fontproperties, angle,
                                   ismath=False)

                exponent, xt, yt, fp = info[1]
                renderer.draw_text(gc, xt, yt, exponent,
                                   fp, angle,
                                   ismath=False)
                return


        if len(self._substrings)>1:
            # embedded mathtext
            thisx, thisy = self.get_transform().xy_tup((self._x, self._y))
            for s,ismath in self._substrings:
                w, h = renderer.get_text_width_height(
                    s, self._fontproperties, ismath)

                renderx, rendery = thisx, thisy
                if renderer.flipy():
                    canvasw, canvash = renderer.get_canvas_width_height()
                    rendery = canvash-rendery

                renderer.draw_text(gc, renderx, rendery, s,
                                   self._fontproperties, angle,
                                   ismath)
                thisx += w


            return
        bbox, info = self._get_layout(renderer)
        trans = self.get_transform()
        if ismath=='TeX':
            canvasw, canvash = renderer.get_canvas_width_height()
            for line, wh, x, y in info:
                x, y = trans.xy_tup((x, y))
                if renderer.flipy():
                    y = canvash-y

                renderer.draw_tex(gc, x, y, line,
                                  self._fontproperties, angle)
            return

        #print 'xy', self._x, self._y, info
        for line, wh, x, y in info:
            x, y = trans.xy_tup((x, y))

            if renderer.flipy():
                canvasw, canvash = renderer.get_canvas_width_height()
                y = canvash-y

            renderer.draw_text(gc, x, y, line,
                               self._fontproperties, angle,
                               ismath=self.is_math_text())

    def get_color(self):
        "Return the color of the text"
        return self._color

    def get_font_properties(self):
        "Return the font object"
        return self._fontproperties

    def get_name(self):
        "Return the font name as string"
        return self._fontproperties.get_family()[-1]  #  temporary hack.

    def get_style(self):
        "Return the font style as string"
        return self._fontproperties.get_style()

    def get_size(self):
        "Return the font size as integer"
        return self._fontproperties.get_size_in_points()

    def get_weight(self):
        "Get the font weight as string"
        return self._fontproperties.get_weight()

    def get_fontname(self):
        'alias for get_name'
        return self._fontproperties.get_family()[-1]  #  temporary hack.

    def get_fontstyle(self):
        'alias for get_style'
        return self._fontproperties.get_style()

    def get_fontsize(self):
        'alias for get_size'
        return self._fontproperties.get_size_in_points()

    def get_fontweight(self):
        'alias for get_weight'
        return self._fontproperties.get_weight()


    def get_ha(self):
        'alias for get_horizontalalignment'
        return self.get_horizontalalignment()

    def get_horizontalalignment(self):
        "Return the horizontal alignment as string"
        return self._horizontalalignment

    def get_position(self):
        "Return x, y as tuple"
        return self._x, self._y

    def get_prop_tup(self):
        """
        Return a hashable tuple of properties

        Not intended to be human readable, but useful for backends who
        want to cache derived information about text (eg layouts) and
        need to know if the text has changed
        """

        return (self._x, self._y, self._text, self._color,
                self._verticalalignment, self._horizontalalignment,
                hash(self._fontproperties), self._rotation,
                self.get_transform().as_vec6_val(),
                )

    def get_text(self):
        "Get the text as string"
        return self._text

    def get_va(self):
        'alias for getverticalalignment'
        return self.get_verticalalignment()

    def get_verticalalignment(self):
        "Return the vertical alignment as string"
        return self._verticalalignment

    def get_window_extent(self, renderer=None):
        #return _unit_box
        if not self.get_visible(): return _unit_box
        if self._text == '':
            tx, ty = self.get_transform().xy_tup( (self._x, self._y) )
            return lbwh_to_bbox(tx,ty,0,0)

        if renderer is not None:
            self._renderer = renderer
        if self._renderer is None:
            raise RuntimeError('Cannot get window extent w/o renderer')

        angle = self.get_rotation()
        if angle==0:
            ismath = self.is_math_text()
            if ismath=='TeX': m = None
            else: m = self._rgxsuper.match(self._text)
            if m is not None:
                bbox, tmp = self._get_layout_super(self._renderer, m)
                return bbox
        bbox, info = self._get_layout(self._renderer)
        return bbox



    def get_rotation_matrix(self, x0, y0):

        theta = pi/180.0*self.get_rotation()
        # translate x0,y0 to origin
        Torigin = array([ [1, 0, -x0],
                           [0, 1, -y0],
                           [0, 0, 1  ]])

        # rotate by theta
        R = array([ [cos(theta),  -sin(theta), 0],
                     [sin(theta), cos(theta), 0],
                     [0,           0,          1]])

        # translate origin back to x0,y0
        Tback = array([ [1, 0, x0],
                         [0, 1, y0],
                         [0, 0, 1  ]])


        return dot(dot(Tback,R), Torigin)

    def set_backgroundcolor(self, color):
        """
        Set the background color of the text by updating the bbox (see set_bbox for more info)

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

    def set_family(self, fontname):
        """
        Set the font family

        ACCEPTS: [ 'serif' | 'sans-serif' | 'cursive' | 'fantasy' | 'monospace' ]
        """
        self._fontproperties.set_family(fontname)

    def set_variant(self, variant):
        """
        Set the font variant, eg,

        ACCEPTS: [ 'normal' | 'small-caps' ]
        """
        self._fontproperties.set_variant(variant)

    def set_name(self, fontname):
        """
        Set the font name,

        ACCEPTS: string eg, ['Sans' | 'Courier' | 'Helvetica' ...]
        """
        self._fontproperties.set_name(fontname)

    def set_fontname(self, fontname):
        'alias for set_name'
        self.set_name(fontname)

    def set_style(self, fontstyle):
        """
        Set the font style

        ACCEPTS: [ 'normal' | 'italic' | 'oblique']
        """
        self._fontproperties.set_style(fontstyle)

    def set_fontstyle(self, fontstyle):
        'alias for set_style'
        self._fontproperties.set_style(fontstyle)

    def set_size(self, fontsize):
        """
        Set the font size, eg, 8, 10, 12, 14...

        ACCEPTS: [ size in points | relative size eg 'smaller', 'x-large' ]
        """
        self._fontproperties.set_size(fontsize)

    def set_fontsize(self, fontsize):
        'alias for set_size'
        self._fontproperties.set_size(fontsize)

    def set_fontweight(self, weight):
        'alias for set_weight'
        self._fontproperties.set_weight(weight)

    def set_weight(self, weight):
        """
        Set the font weight

        ACCEPTS: [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
        """
        self._fontproperties.set_weight(weight)

    def set_position(self, xy):
        """
        Set the xy position of the text

        ACCEPTS: (x,y)
        """
        self.set_x(xy[0])
        self.set_y(xy[1])

    def set_x(self, x):
        """
        Set the x position of the text

        ACCEPTS: float
        """
        self._x = float(x)


    def set_y(self, y):
        """
        Set the y position of the text

        ACCEPTS: float
        """
        self._y = float(y)


    def set_rotation(self, s):
        """
        Set the rotation of the text

        ACCEPTS: [ angle in degrees 'vertical' | 'horizontal'
        """
        self._rotation = s



    def set_va(self, align):
        'alias for set_verticalalignment'
        self.set_verticalalignment(align)

    def set_verticalalignment(self, align):
        """
        Set the vertical alignment

        ACCEPTS: [ 'center' | 'top' | 'bottom' ]
        """
        legal = ('top', 'bottom', 'center')
        if align not in legal:
            raise ValueError('Vertical alignment must be one of %s' % str(legal))

        self._verticalalignment = align

    def set_text(self, s):
        """
        Set the text string s

        ACCEPTS: string
        """
        if not is_string_like(s):
            raise TypeError("This doesn't look like a string: '%s'"%s)
        self._text = s
        #self._substrings = scanner(s)  # support embedded mathtext
        self._substrings = []           # ignore embedded mathtext for now

    def is_math_text(self):
        if rcParams['text.usetex']: return 'TeX'
        if not matplotlib._havemath: return False
        if len(self._text)<2: return False
        return ( self._text.startswith('$') and
                 self._text.endswith('$') )

    def set_fontproperties(self, fp):
        """
        Set the font properties that control the text

        ACCEPTS: a matplotlib.font_manager.FontProperties instance
        """
        self._fontproperties = fp




    def _get_layout_super(self, renderer, m):
        """
        a special case optimization if a log super and angle = 0
        Basically, mathtext is slow and we can do simple superscript layout "by hand"
        """

        key = self.get_prop_tup()
        if self.cached.has_key(key): return self.cached[key]

        base, exponent = m.group(1), m.group(2)
        size =  self._fontproperties.get_size_in_points()
        fpexp = self._fontproperties.copy()
        fpexp.set_size(0.7*size)
        wb,hb = renderer.get_text_width_height(base, self._fontproperties, False)
        we,he = renderer.get_text_width_height(exponent, fpexp, False)

        w = wb+we

        xb, yb = self.get_transform().xy_tup((self._x, self._y))
        xe = xb+1.1*wb
        ye = yb+0.5*hb
        h = ye+he-yb




        if self._horizontalalignment=='center':  xo = -w/2.
        elif self._horizontalalignment=='right':  xo = -w
        else: xo = 0
        if self._verticalalignment=='center':    yo = -hb/2.
        elif self._verticalalignment=='top':  yo = -hb
        else: yo = 0

        xb += xo
        yb += yo
        xe += xo
        ye += yo
        bbox = lbwh_to_bbox(xb, yb, w, h)

        if renderer.flipy():
            canvasw, canvash = renderer.get_canvas_width_height()
            yb = canvash-yb
            ye = canvash-ye


        val = ( bbox, ((base, xb, yb), (exponent, xe, ye, fpexp)))
        self.cached[key] = val

        return val



class TextWithDash(Text):
    """
    This is basically a Text with a dash (drawn with a Line2D)
    before/after it. It is intended to be a drop-in replacement
    for Text, and should behave identically to Text when
    dashlength=0.0.

    The dash always comes between the point specified by
    set_position() and the text. When a dash exists, the
    text alignment arguments (horizontalalignment,
    verticalalignment) are ignored.

    dashlength is the length of the dash in canvas units.
    (default=0.0).

    dashdirection is one of 0 or 1, where 0 draws the dash
    after the text and 1 before.
    (default=0).

    dashrotation specifies the rotation of the dash, and
    should generally stay None. In this case
    self.get_dashrotation() returns self.get_rotation().
    (I.e., the dash takes its rotation from the text's
    rotation). Because the text center is projected onto
    the dash, major deviations in the rotation cause
    what may be considered visually unappealing results.
    (default=None).

    dashpad is a padding length to add (or subtract) space
    between the text and the dash, in canvas units.
    (default=3).

    dashpush "pushes" the dash and text away from the point
    specified by set_position() by the amount in canvas units.
    (default=0)

    NOTE: The alignment of the two objects is based on the
    bbox of the Text, as obtained by get_window_extent().
    This, in turn, appears to depend on the font metrics
    as given by the rendering backend. Hence the quality
    of the "centering" of the label text with respect to
    the dash varies depending on the backend used.

    NOTE2: I'm not sure that I got the get_window_extent()
    right, or whether that's sufficient for providing the
    object bbox.
    """
    __name__ = 'textwithdash'

    def __init__(self,
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='center',
                 horizontalalignment='center',
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 dashlength=0.0,
                 dashdirection=0,
                 dashrotation=None,
                 dashpad=3,
                 dashpush=0,
                 xaxis=True,
                 ):

        Text.__init__(self, x=x, y=y, text=text, color=color,
                      verticalalignment=verticalalignment,
                      horizontalalignment=horizontalalignment,
                      multialignment=multialignment,
                      fontproperties=fontproperties, rotation=rotation)

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

    def draw(self, renderer):
        self.update_coords(renderer)
        Text.draw(self, renderer)
        if self.get_dashlength() > 0.0:
            self.dashline.draw(renderer)

    def update_coords(self, renderer):
        """Computes the actual x,y coordinates for
        text based on the input x,y and the
        dashlength. Since the rotation is with respect
        to the actual canvas's coordinates we need to
        map back and forth.
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
        theta = pi*(angle/180.0+dashdirection-1)
        cos_theta, sin_theta = cos(theta), sin(theta)

        transform = self.get_transform()

        # Compute the dash end points
        # The 'c' prefix is for canvas coordinates
        cxy = array(transform.xy_tup((dashx, dashy)))
        cd = array([cos_theta, sin_theta])
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
        we = Text.get_window_extent(self, renderer=renderer)
        w, h = we.width(), we.height()
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
        cwd = array([dx, dy])/2
        cwd *= 1+dashpad/sqrt(dot(cwd,cwd))
        cw = c2+(dashdirection*2-1)*cwd

        self._x, self._y = transform.inverse_xy_tup(tuple(cw))

        # Now set the window extent
        # I'm not at all sure this is the right way to do this.
        we = Text.get_window_extent(self, renderer=renderer)
        self._twd_window_extent = we.deepcopy()
        self._twd_window_extent.update(((c1[0], c1[1]),), False)

        # Finally, make text align center
        Text.set_horizontalalignment(self, 'center')
        Text.set_verticalalignment(self, 'center')

    def get_window_extent(self, renderer=None):
        self.update_coords(renderer)
        if self.get_dashlength() == 0.0:
            return Text.get_window_extent(self, renderer=renderer)
        else:
            return self._twd_window_extent

    def get_dashlength(self):
        return self._dashlength

    def set_dashlength(self, dl):
        """
        Set the length of the dash.

        ACCEPTS: float
        """
        self._dashlength = dl

    def get_dashdirection(self):
        return self._dashdirection

    def set_dashdirection(self, dd):
        """
        Set the direction of the dash following the text.
        1 is before the text and 0 is after. The default
        is 0, which is what you'd want for the typical
        case of ticks below and on the left of the figure.

        ACCEPTS: int
        """
        self._dashdirection = dd

    def get_dashrotation(self):
        if self._dashrotation == None:
            return self.get_rotation()
        else:
            return self._dashrotation

    def set_dashrotation(self, dr):
        """
        Set the rotation of the dash.

        ACCEPTS: float
        """
        self._dashrotation = dr

    def get_dashpad(self):
        return self._dashpad

    def set_dashpad(self, dp):
        """
        Set the "pad" of the TextWithDash, which
        is the extra spacing between the dash and
        the text, in canvas units.

        ACCEPTS: float
        """
        self._dashpad = dp

    def get_dashpush(self):
        return self._dashpush

    def set_dashpush(self, dp):
        """
        Set the "push" of the TextWithDash, which
        is the extra spacing between the beginning
        of the dash and the specified position.

        ACCEPTS: float
        """
        self._dashpush = dp

    def get_position(self):
        "Return x, y as tuple"
        return self._dashx, self._dashy

    def set_position(self, xy):
        """
        Set the xy position of the TextWithDash.

        ACCEPTS: (x,y)
        """
        self.set_x(xy[0])
        self.set_y(xy[1])

    def set_x(self, x):
        """
        Set the x position of the TextWithDash.

        ACCEPTS: float
        """
        self._dashx = float(x)

    def set_y(self, y):
        """
        Set the y position of the TextWithDash.

        ACCEPTS: float
        """
        self._dashy = float(y)

    def set_transform(self, t):
        """
        Set the Transformation instance used by this artist.

        ACCEPTS: a matplotlib.transform transformation instance
        """
        Text.set_transform(self, t)
        self.dashline.set_transform(t)

    def get_figure(self):
        'return the figure instance'
        return self.figure

    def set_figure(self, fig):
        """
        Set the figure instance the artist belong to.

        ACCEPTS: a matplotlib.figure.Figure instance
        """
        Text.set_figure(self, fig)
        self.dashline.set_figure(fig)

class _Annotation(Text):
    """
    A Text class to make annotating things in the figure: Figure,
    Axes, Point, Rectangle, etc... easier
    """
    def __init__(self, artist, s, loc=None,
                 padx='auto', pady='auto', autopad=3,
                 lineprops=None,
                 coords=None,
                 **props):
        """
        Annotate the matplotlib.Artist artist with string s.  kwargs
        props are passed on to the Text base class and are text
        properties.

        loc is an x, y tuple.  If the location codes are a string and
        the artist supports the
        "get_window_extent method" (eg matplotlib.patches.Patch and
        children, Text, Axes, Figure, Line2D) the location code can be
        a pair of strings.  Here are a few examples

          A: 'inside left', 'inside upper'
          B: 'outside right', 'outside lower'
          C: 'center', 'center'
          D: 'inside left', 'outside bottom'
          E: 'center', 'outside top'

          inside and outside cannot be used with 'center'.  With
          upper, lower, left and right, inside will be assumed if
          inside|outside is not provided

                                  E
             --------------------------------------------
             | A                                        |
             |                                          |
             |                                          |
             |                     C                    |
             |                                          |
             |                                          |
             |                                          |
             |__________________________________________|
              D                                          B

        These codes also work with Axes and Figure instances
        Otherwise it must be an x,y pair which will use the artist's
        own transformation

        eg
        Annotation(rectangle, 'some text', loc=('center', 'outside top'), color='red', size=14)

        Annotation(axes, 'A', loc=('inside left', 'inside top'))

        padx and pady are number of points to pad the text in the x
        and y direction.  When used with string codes, 'auto' will pad
        autopad points in the appropriate direction given the
        inside/outside left/right/center bottom/top/center location
        codes

        lineprops, if not None, is a dictionary of line properties
        used to draw a line between the annotation and the point being
        annotated (if lineprops is None, no line is drawn).  The keys
        of the dictionary are line properties (eg linewidth, color,
        linestyle -- see matplotlib.lines for more information).  In
        addition, the following dictionary key/value pairs are
        supported for the lineprops

            shrink : the value in points that will be used to shorten
                     the line on each end
            xalign : left | right | center | auto - where to align the line on the text
            yalign : bottom | top | center | auto - where to align the line on the text

        Here is an example with xalign='center' and yalign='bottom'

            ------------------------
            |                      |
            |  the text annotation |
            |______________________|
                                              <---shrink shortens the line here
                       /
                      /
                     /
                    /
                                               <---and here
                  loc


         coords, if not None, is a string that will specify the
         coordinate system of the x,y location.  Possible choices are

           'figure points'   : points from the lower left corner of the figure
           'figure pixels'   : pixels from the lower left corner of the figure                           'figure fraction' : 0,0 is lower left of figure and 1,1 is upper, right
           'axes points'     : points from lower left corner of axes
           'axes pixels'     : pixels from lower left corner of axes
           'axes fraction'   : 0,1 is lower left of axes and 1,1 is upper right
           'data'            : use the coordinate system of the object being annotated (default)
           'polar'           : you can specify theta, r for the annotation, even
                               in cartesian plots.  Note that if you
                               are using a polar axes, you do not need
                               to specify polar for the coordinate
                               system since that is the native"data" coordinate system.

        If a points or pixels option is specified, values will be
        added to the left, bottom and if negative, values will be
        subtracted from the top, right.  Eg,

          # 10 points to the right of the left border of the axes and
          # 5 points below the top border
          loc=(10,-5), coords='axes points'


        """

        # we'll draw ourself after the artist we annotate by default
        zorder = props.get('zorder', artist.get_zorder() + 1)
        Text.__init__(self, text=s, **props)

        self.line = Line2D([0], [0])
        self._shrink = 0.
        self.set_lineprops(lineprops)
        self.set_zorder(zorder)
        self.set_transform(identity_transform())
        self._loc = tuple(loc)
        self._coords = coords
        self._padx, self._pady, self._autopad = padx, pady, autopad
        self._annotateArtist = artist
        # funcx and funcy  are used to place the x, y coords for
        # artists who define get_window_extent

        xloc, yloc = self._loc
        self._process_xloc(xloc)
        self._process_yloc(yloc)

        self._renderer = None

    def set_lineprops(self, lineprops):
        """
        Set the c padding in points
        ACCEPTS: float value in points or the string 'auto'
        """
        self._lineprops = lineprops
        if lineprops is not None:
            lineprops = lineprops.copy()
            self._shrink = lineprops.pop('shrink', 0.)
            self._xalign = lineprops.pop('xalign', 'auto')
            self._yalign = lineprops.pop('yalign', 'auto')
            self.line.update(lineprops)

    def get_lineprops(self):
        'get the x padding in points'
        return self._lineprops

    def set_padx(self, padx):
        """
        Set the c padding in points
        ACCEPTS: float value in points or the string 'auto'
        """
        self._padx = padx
        self._process_xloc(self._loc[0])

    def get_padx(self):
        'get the x padding in points'
        return self._padx

    def set_pady(self, pady):
        """
        Set the y padding in points
        ACCEPTS: float value in points or the string 'auto'
        """
        self._pady = pady
        self._process_yloc(self._loc[1])

    def get_pady(self):
        'get the y padding in points'
        return self._pady

    def set_autopad(self, autopad):
        """
        Set the y padding in points
        ACCEPTS: float value in points
        """
        self._autopad = autopad
        self._process_xloc(self._loc[0])
        self._process_yloc(self._loc[1])

    def get_autopad(self):
        'get the y padding in points'
        return self._autopad

    def _process_xloc(self, xloc):
        """
        This function will set the horiz and vertical alignment
        properties, and set the attr _funcx to place the x coord at
        draw time
        """

        props = dict()
        if is_numlike(xloc):
            if self._padx=='auto':
                self._padx = 0.
            self._funcx = None
            return
        if not is_string_like(xloc):
            raise ValueError('x location code must be a number or string')
        xloc = xloc.lower().strip()

        if xloc=='center':
            props['horizontalalignment'] = 'center'
            def funcx(left, right):
                return 0.5*(left + right)
            if self._padx=='auto':
                self._padx = 0.
        else:
            tup = xloc.split(' ')
            if len(tup)!=2:
                raise ValueError('location code looks like "inside|outside left|right".  You supplied "%s"'%xloc)

            inout, leftright = tup

            if inout not in ('inside', 'outside'):
                raise ValueError('x in/out: bad location code "%s"'%xloc)
            if leftright not in ('left', 'right'):
                raise ValueError('x left/right: bad location code "%s"'%xloc)
            if inout=='inside' and leftright=='left':
                props['horizontalalignment'] = 'left'
                def funcx(left, right):
                    return left
                if self._padx=='auto':
                    self._padx = self._autopad
            elif inout=='inside' and leftright=='right':
                props['horizontalalignment'] = 'right'
                def funcx(left, right):
                    return right
                if self._padx=='auto':
                    self._padx = -self._autopad
            elif inout=='outside' and leftright=='left':
                props['horizontalalignment'] = 'right'
                def funcx(left, right):
                    return left
                if self._padx=='auto':
                    self._padx = -self._autopad
            elif inout=='outside' and leftright=='right':
                props['horizontalalignment'] = 'left'
                def funcx(left, right):
                    return right
                if self._padx=='auto':
                    self._padx = self._autopad

        self.update(props)
        self._funcx = funcx

    def _process_yloc(self, yloc):
        """
        This function will set the horiz and vertical alignment
        properties, and set the attr _funcy to place the y coord at
        draw time
        """
        props = dict()
        if is_numlike(yloc):
            if self._pady=='auto':
                self._pady = 0.
            self._funcy = None
            return # nothing to do

        if not is_string_like(yloc):
            raise ValueError('y location code must be a number or string')
        yloc = yloc.lower().strip()

        if yloc=='center':
            props['verticalalignment'] = 'center'
            def funcy(bottom, top):
                return 0.5*(bottom + top)
            if self._pady=='auto':
                self._pady = 0.

        else:
            tup = yloc.split(' ')
            if len(tup)!=2:
                raise ValueError('location code looks like "inside|outside bottom|top".  You supplied "%s"'%yloc)

            inout, bottomtop = tup

            if inout not in ('inside', 'outside'):
                raise ValueError('y in/out: bad location code "%s"'%yloc)
            if bottomtop not in ('bottom', 'top'):
                raise ValueError('y bottom/top: bad location code "%s"'%yloc)
            if inout=='inside' and bottomtop=='bottom':
                props['verticalalignment'] = 'bottom'
                def funcy(bottom, top):
                    return bottom
                if self._pady=='auto':
                    self._pady = self._autopad
            elif inout=='inside' and bottomtop=='top':
                props['verticalalignment'] = 'top'
                def funcy(bottom, top):
                    return top
                if self._pady=='auto':
                    self._pady = -self._autopad
            elif inout=='outside' and bottomtop=='bottom':
                props['verticalalignment'] = 'top'
                def funcy(bottom, top):
                    return bottom
                if self._pady=='auto':
                    self._pady = -self._autopad
            elif inout=='outside' and bottomtop=='top':
                props['verticalalignment'] = 'bottom'
                def funcy(bottom, top):
                    return top
                if self._pady=='auto':
                    self._pady = self._autopad

        self.update(props)
        self._funcy = funcy

    def update_positions(self, renderer=None):
        if renderer is None and self._renderer is None:
            raise RuntimeError('renderer not set')
        if renderer is None:
            renderer = self._renderer
        if self._funcx is not None and self._funcy is not None:
            extent = getattr(self._annotateArtist, 'get_window_extent')
            bbox = extent(renderer)
            l,b,w,h = bbox.get_bounds()
            r = l+w
            t = b+h
            self._x = self._funcx(l,r)
            self._y = self._funcy(b,t)
        else:
            if self._coords is None or self._coords=='data':
                trans = self._annotateArtist.get_transform()
                self._x, self._y = trans.xy_tup(self._loc)
            elif self._coords=='polar':
                theta, r = self._loc
                x = r*cos(theta)
                y = r*sin(theta)
                trans = self._annotateArtist.get_transform()
                self._x, self._y = trans.xy_tup((x,y))
            elif self._coords=='figure points':
                #points from the lower left corner of the figure
                dpi = self.figure.dpi.get()
                l,b,w,h = self.figure.bbox.get_bounds()
                r = l+w
                t = b+h

                x, y = self._loc
                x *= dpi/72.
                y *= dpi/72.
                if x<0:
                    self._x = r + x
                else:
                    self._x = x
                if y<0:
                    self._y = t + y
                else:
                    self._y = y


            elif self._coords=='figure pixels':
                #pixels from the lower left corner of the figure
                l,b,w,h = self.figure.bbox.get_bounds()
                r = l+w
                t = b+h
                x, y = self._loc
                if x<0:
                    self._x = r + x
                else:
                    self._x = x
                if y<0:
                    self._y = t + y
                else:
                    self._y = y
            elif self._coords=='figure fraction':
                #(0,0) is lower left, (1,1) is upper right of figure
                trans = self.figure.transFigure
                self._x, self._y = trans.xy_tup(self._loc)
            elif self._coords=='axes points':
                #points from the lower left corner of the axes
                dpi = self.figure.dpi.get()
                x, y = self._loc
                l,b,w,h = self._annotateArtist.axes.bbox.get_bounds()
                r = l+w
                t = b+h
                if x<0:
                    self._x = r + x*dpi/72.
                else:
                    self._x = l + x*dpi/72.
                if y<0:
                    self._y = t + y*dpi/72.
                else:
                    self._y = b + y*dpi/72.
            elif self._coords=='axes pixels':
                #pixels from the lower left corner of the axes
                x, y = self._loc
                l,b,w,h = self._annotateArtist.axes.bbox.get_bounds()
                r = l+w
                t = b+h
                if x<0:
                    self._x = r + x
                else:
                    self._x = l + x
                if y<0:
                    self._y = t + y
                else:
                    self._y = b + y
            elif self._coords=='axes fraction':
                #(0,0) is lower left, (1,1) is upper right of axes
                trans = self._annotateArtist.transAxes
                self._x, self._y = trans.xy_tup(self._loc)

        dpi = self.figure.dpi.get()
        dx = self._padx * dpi/72.
        dy = self._pady * dpi/72.
        self._x += dx
        self._y += dy

    def draw(self, renderer):
        if renderer is not None:
            self._renderer = renderer
        self.update_positions()
        #print 'drawing annotation', self._x, self._y, self._text
        Text.draw(self, renderer)
        if self._lineprops is not None:
            l,b,w,h = self.get_window_extent(renderer).get_bounds()
            dpi = self.figure.dpi.get()
            dx = self._padx * dpi/72.
            dy = self._pady * dpi/72.
            x0, y0 = self._x - dx, self._y - dy
            r = l+w
            t = b+h
            xc = 0.5*(l+r)
            yc = 0.5*(b+t)
            # pick the x,y corner of the text bbox closest to point
            # annotated
            if self._xalign=='left': x = l
            elif self._xalign=='right': x = r
            elif self._xalign=='center': x = xc
            else:
                dsu = [(abs(val-x0), val) for val in l, r, xc]
                dsu.sort()
                d, x = dsu[0]

            if self._yalign=='bottom': y = b
            elif self._yalign=='top': y = t
            elif self._yalign=='center': y = yc
            else:
                dsu = [(abs(val-y0), val) for val in b, t, yc]
                dsu.sort()
                d, y = dsu[0]



            if self._shrink:
                r = math.sqrt((x-x0)**2 + (y-y0)**2)
                theta = math.atan2(y-y0,x-x0)
                dx = self._shrink*dpi/72.*math.cos(theta)
                dy = self._shrink*dpi/72.*math.sin(theta)
                x0 += dx
                x -= dx
                y0 += dy
                y -= dy

            self.line.set_data([x0, x], [y0, y])
            self.line.draw(renderer)

class Annotation(Text):
    """
    A Text class to make annotating things in the figure: Figure,
    Axes, Point, Rectangle, etc... easier
    """
    def __init__(self, s, xy,
                 xycoords='data',
                 xytext=None,
                 textcoords=None,
                 arrowprops=None,
                 **kwargs):
        """
        Annotate the x,y point xy with text s at x,y location xytext
        (xytext if None defaults to xy and textcoords if None defaults
        to xycoords).

        arrowprops, if not None, is a dictionary of line properties
        (see matplotlib.lines.Line2D) for the arrow that connects
        annotation to the point.   Valid keys are

          - width : the width of the arrow in points
          - frac  : the fraction of the arrow length occupied by the head
          - headwidth : the width of the base of the arrow head in points
          - shrink: often times it is convenient to have the arrowtip
            and base a bit away from the text and point being
            annotated.  If d is the distance between the text and
            annotated point, shrink will shorten the arrow so the tip
            and base are shink percent of the distance d away from the
            endpoints.  ie, shrink=0.05 is 5%
          - any key for matplotlib.patches.polygon

        xycoords and textcoords are a string that indicates the
        coordinates of xy and xytext.

           'figure points'   : points from the lower left corner of the figure
           'figure pixels'   : pixels from the lower left corner of the figure                           'figure fraction' : 0,0 is lower left of figure and 1,1 is upper, right
           'axes points'     : points from lower left corner of axes
           'axes pixels'     : pixels from lower left corner of axes
           'axes fraction'   : 0,1 is lower left of axes and 1,1 is upper right
           'data'            : use the coordinate system of the object being annotated (default)
           'polar'           : you can specify theta, r for the annotation, even
                               in cartesian plots.  Note that if you
                               are using a polar axes, you do not need
                               to specify polar for the coordinate
                               system since that is the native"data" coordinate system.

        If a points or pixels option is specified, values will be
        added to the left, bottom and if negative, values will be
        subtracted from the top, right.  Eg,

          # 10 points to the right of the left border of the axes and
          # 5 points below the top border
          xy=(10,-5), xycoords='axes points'


        """
        if xytext is None:
            xytext = xy
        if textcoords is None:
            textcoords = xycoords
        # we'll draw ourself after the artist we annotate by default
        x,y = self.xytext = xytext
        Text.__init__(self, x, y, s, **kwargs)
        self.xy = xy
        self.arrowprops = arrowprops
        self.arrow = None
        self.xycoords = xycoords
        self.textcoords = textcoords

    def _get_xy(self, x, y, s):
        if s=='data':
            trans = self.axes.transData
            return trans.xy_tup((x,y))
        elif s=='polar':
            theta, r = x, y
            x = r*cos(theta)
            y = r*sin(theta)
            trans = self.axes.transData
            return trans.xy_tup((x,y))
        elif s=='figure points':
            #points from the lower left corner of the figure
            dpi = self.figure.dpi.get()
            l,b,w,h = self.figure.bbox.get_bounds()
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
            l,b,w,h = self.figure.bbox.get_bounds()
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
            return trans.xy_tup((x,y))
        elif s=='axes points':
            #points from the lower left corner of the axes
            dpi = self.figure.dpi.get()
            l,b,w,h = self.axes.bbox.get_bounds()
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

            l,b,w,h = self.axes.bbox.get_bounds()
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
            return trans.xy_tup((x,y))


    def update_positions(self, renderer):

        x, y = self.xytext
        self._x, self._y = self._get_xy(x, y, self.textcoords)


        x, y = self.xy
        x, y = self._get_xy(x, y, self.xycoords)

        if self.arrowprops:
            x0, y0 = x, y
            l,b,w,h = self.get_window_extent(renderer).get_bounds()
            dpi = self.figure.dpi.get()
            r = l+w
            t = b+h
            xc = 0.5*(l+r)
            yc = 0.5*(b+t)
            # pick the x,y corner of the text bbox closest to point
            # annotated
            dsu = [(abs(val-x0), val) for val in l, r, xc]
            dsu.sort()
            d, x = dsu[0]

            dsu = [(abs(val-y0), val) for val in b, t, yc]
            dsu.sort()
            d, y = dsu[0]


            d = self.arrowprops.copy()
            width = popd(d, 'width', 4)
            headwidth = popd(d, 'headwidth', 12)
            frac = popd(d, 'frac', 0.1)
            shrink = popd(d, 'shrink', 0.0)


            theta = math.atan2(y-y0, x-x0)
            r = math.sqrt((y-y0)**2. + (x-x0)**2.)
            dx = shrink*r*math.cos(theta)
            dy = shrink*r*math.sin(theta)

            self.arrow = YAArrow(self.figure.dpi, (x0+dx,y0+dy), (x-dx, y-dy),
                            width=width, headwidth=headwidth, frac=frac,
                            **d)

    def draw(self, renderer):
        self.update_positions(renderer)

        if self.arrow is not None:
            self.arrow.draw(renderer)

        Text.draw(self, renderer)


artist.kwdocd['Text'] = artist.kwdoc(Text)
artist.kwdocd['TextWithDash'] = artist.kwdoc(TextWithDash)
artist.kwdocd['Annotation'] = artist.kwdoc(Annotation)
