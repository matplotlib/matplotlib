"""
Figure and Axes text
"""
from __future__ import division
from matplotlib import verbose
import matplotlib
import math
from artist import Artist
from cbook import enumerate, popd
from font_manager import FontProperties
from matplotlib import rcParams
from patches import bbox_artist
from numerix import sin, cos, pi, Matrix, cumsum
from transforms import lbwh_to_bbox, bbox_all

def _process_text_args(override, fontdict=None, **kwargs):
    "Return an override dict.  See 'text' docstring for info"

    if fontdict is not None:
        override.update(fontdict)

    override.update(kwargs)
    return override




class Text(Artist):
    """
    Handle storing and drawing of text in window or data coordinates

    """

    zorder = 3
    def __init__(self, 
                 x=0, y=0, text='',
                 color=None,          # defaults to rc params
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 multialignment=None,
                 fontproperties=None, # defaults to FontProperties()
                 rotation=None,
                 ):

        Artist.__init__(self)
        self.cached = {}
        self._x, self._y = x, y

        if color is None: color = rcParams['text.color']
        if fontproperties is None: fontproperties=FontProperties()

        self._color = color
        self._text = text
        self._verticalalignment = verticalalignment
        self._horizontalalignment = horizontalalignment
        self._multialignment = multialignment        
        self._rotation = rotation
        self._fontproperties = fontproperties
        self._bbox = None
        
    def _get_multialignment(self):
        if self._multialignment is not None: return self._multialignment
        else: return self._horizontalalignment
    
    def get_rotation(self):
        'return the text angle as float'
        #return 0
        if self._rotation in ('horizontal', None):
            angle = 0.
        elif self._rotation == 'vertical':            
            angle = 90.
        else:
            angle = float(self._rotation)
        return angle%360
    
    def copy_properties(self, t):
        'Copy properties from t to self'

        self._color = t._color
        self._multialignment = t._multialignment        
        self._verticalalignment = t._verticalalignment
        self._horizontalalignment = t._horizontalalignment
        self._fontproperties = t._fontproperties.copy()
        self._rotation = t._rotation


    def _get_layout(self, renderer):

        # layout the xylocs in display coords as if angle = zero and
        # then rotate them around self._x, self._y
        
        key = self.get_prop_tup()
        if self.cached.has_key(key): return self.cached[key]
        
        horizLayout = []
        pad =2
        thisx, thisy = self._transform.xy_tup( (self._x, self._y) )
        width = 0
        height = 0

        xmin, ymin = thisx, thisy
        if self.is_math_text():
            lines = [self._text]
        else:
            lines = self._text.split('\n')

        whs = []
        for line in lines:
            w,h = renderer.get_text_width_height(
                line, self._fontproperties, ismath=self.is_math_text())
            if not len(line) and not self.is_math_text():
                # approx the height of empty line with tall char
                tmp, h = renderer.get_text_width_height(
                'T', self._fontproperties, ismath=False)

            whs.append( (w,h) )
            offsety = h+pad
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
        cornersRotated = [M*Matrix([[thisx],[thisy],[1]]) for thisx, thisy in cornersHoriz]

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
        tx, ty = self._transform.xy_tup( (self._x, self._y) )
        
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
        xys = [M*Matrix([[thisx],[thisy],[1]]) for thisx, thisy in offsetLayout]


        tx = [float(v[0][0])+offsetx for v in xys]
        ty = [float(v[1][0])+offsety for v in xys]

        # now inverse transform back to data coords
        xys = [self._transform.inverse_xy_tup( xy ) for xy in zip(tx, ty)]

        xs, ys = zip(*xys)

        ret = bbox, zip(lines, whs, xs, ys)
        self.cached[key] = ret
        return ret
         

    def set_bbox(self, rectprops):
        """
Draw a bounding box around self.  rect props are any settable
properties for a rectangle, eg color='r', alpha=0.5

ACCEPTS: rectangle prop dict plus key 'pad' which is a pad in points
        """
        self._bbox = rectprops

    def draw(self, renderer):
        if self._text=='': return

        gc = renderer.new_gc()
        gc.set_foreground(self._color)
        gc.set_alpha(self._alpha)
        if self.get_clip_on():
            gc.set_clip_rectangle(self.clipbox.get_bounds())


            
        if self._bbox:
            bbox_artist(self, renderer, self._bbox)
        angle = self.get_rotation()
        bbox, info = self._get_layout(renderer)

        for line, wh, x, y in info:
            x, y = self._transform.xy_tup((x, y))
            #renderer.draw_arc(gc, (1,0,0),
            #                  x, y, 2, 2, 0.0, 360.0)

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

        x, y = self._transform.xy_tup((self._x, self._y))
        return (x, y, self._text, self._color,
                self._verticalalignment, self._horizontalalignment,
                hash(self._fontproperties), self._rotation)

    def get_text(self):
        "Get the text as string"
        return self._text

    def get_va(self):
        'alias for getverticalalignment'
        return self.get_verticalalignment()

    def get_verticalalignment(self):
        "Return the vertical alignment as string"
        return self._verticalalignment

    def get_window_extent(self, renderer):
        bbox, info = self._get_layout(renderer)
        return bbox

            

    def get_rotation_matrix(self, x0, y0):

        theta = -pi/180.0*self.get_rotation()
        # translate x0,y0 to origin
        Torigin = Matrix([ [1, 0, -x0],
                           [0, 1, -y0],
                           [0, 0, 1  ]]) 
        
        # rotate by theta
        R = Matrix([ [cos(theta),  sin(theta), 0],
                     [-sin(theta), cos(theta), 0],
                     [0,           0,          1]]) 

        # translate origin back to x0,y0
        Tback = Matrix([ [1, 0, x0],
                         [0, 1, y0],
                         [0, 0, 1  ]]) 


        return Tback*R*Torigin
        
    def set_backgroundcolor(self, color):
        """
Set the background color of the text

ACCEPTS: any matplotlib color - see help(colors)
"""
        self._backgroundcolor = color

        
    def set_color(self, color):
        """
Set the foreground color of the text

ACCEPTS: any matplotlib color - see help(colors)
"""
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
        try: self._x.set(x)
        except AttributeError: self._x = x


    def set_y(self, y):
        """
Set the y position of the text

ACCEPTS: float
"""
        try: self._y.set(y)
        except AttributeError: self._y = y
        
        
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
        self._text = s




    def is_math_text(self):
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

        
        
