"""
A PostScript backend, which can produce both PostScript .ps and
Encapsulated PostScript .eps files.
"""

from __future__ import division
import sys, os, time
from cStringIO import StringIO
from matplotlib import verbose, __version__
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase

from matplotlib.cbook import is_string_like
from matplotlib.figure import Figure

from matplotlib.font_manager import fontManager
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import math_parse_s_ps, bakoma_fonts
from matplotlib.text import Text

from matplotlib import get_data_path

from matplotlib.numerix import fromstring, UInt8, Float32, equal, alltrue
import binascii
import re

backend_version = 'Level II'
defaultPaperSize = 8.5,11               # TODO: make this configurable
debugPS = 0




def _num_to_str(val):
    if is_string_like(val): return val
    ival = int(val)
    if val==ival: return str(ival)
    s = "%1.3f"%val
    s = s.rstrip("0")
    s = s.rstrip(".")
    return s

def _nums_to_str(*args):
    return ' '.join(map(_num_to_str,args))

def quote_ps_string(s):
    "Quote dangerous characters of S for use in a PostScript string constant."
    s=s.replace("\\", "\\\\")
    s=s.replace("(", "\\(")
    s=s.replace(")", "\\)")
    s=re.sub(r"[^ -~\n]", lambda x: r"\%03o"%ord(x.group()), s)
    return s


_fontd = {}
_type42 = []


def seq_allequal(seq1, seq2):
    """
    seq1 and seq2 are either None or sequences or numerix arrays
    Return True if both are None or both are seqs with identical
    elements
    """
    if seq1 is None:
        return seq2 is None

    if seq2 is None:
        return False
    #ok, neither are None:, assuming iterable
        
    if len(seq1) != len(seq2): return False
    return alltrue(equal(seq1, seq2))
    
    
class RendererPS(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    def __init__(self, width, height, pswriter):
        self.width = width
        self.height = height
        self._pswriter = pswriter

        # current renderer state (None=uninitialised)
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None

    def set_color(self, r, g, b, store=1):
        if (r,g,b) != self.color:
            if r==g and r==b:
                self._pswriter.write("%1.3f setgray\n"%r)
            else:
                self._pswriter.write("%1.3f %1.3f %1.3f setrgbcolor\n"%(r,g,b))
            self.color = (r,g,b)

    def set_linewidth(self, linewidth):
        if linewidth != self.linewidth:
            self._pswriter.write("%1.3f setlinewidth\n"%linewidth)
            self.linewidth = linewidth

    def set_linejoin(self, linejoin):
        if linejoin != self.linejoin:
            self._pswriter.write("%d setlinejoin\n"%linejoin)
            self.linejoin = linejoin

    def set_linecap(self, linecap):
        if linecap != self.linecap:
            self._pswriter.write("%d setlinecap\n"%linecap)
            self.linecap = linecap

    def set_linedash(self, offset, seq):        
        if self.linedash is not None:
            oldo, oldseq = self.linedash
            if seq_allequal(seq, oldseq): return
            
        if seq is not None and len(seq):
            s="[%s] %d setdash\n"%(_nums_to_str(*seq), offset)
            self._pswriter.write(s)
        else:
            self._pswriter.write("[] 0 setdash\n")
        self.linedash = (offset,seq)

    def set_font(self, fontname, fontsize):
        if (fontname,fontsize) != (self.fontname,self.fontsize):
            out = ("/%s findfont\n"
                   "%1.3f scalefont\n"
                   "setfont\n" % (fontname,fontsize))
            self._pswriter.write(out)
            self.fontname = fontname
            self.fontsize = fontsize

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height

    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop
        """
        if ismath:
            width, height, pswriter = math_parse_s_ps(
                s, 72, prop.get_size_in_points())
            return width, height

        font = self._get_font(prop)
        font.set_text(s, 0.0)
        w, h = font.get_width_height()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        return w, h

    def flipy(self):
        'return true if small y numbers are top for renderer'
        return False

    def _get_font(self, prop):
        key = hash(prop)
        font = _fontd.get(key)
        if font is None:
            fname = fontManager.findfont(prop)

            font = FT2Font(str(fname))
            _fontd[key] = font
            if fname not in _type42:
                _type42.append(fname)
        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, 72.0)
        return font
        
    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0

        If gcFace is not None, fill the arc slice with it.  gcEdge
        is a GraphicsContext instance
        """
        ps = '%s ellipse' % _nums_to_str(angle1, angle2,
                                         0.5*width, 0.5*height, x, y)
        self._draw_ps(ps, gc, rgbFace, "arc")

    def _rgba(self, im, flipud):
        return im.as_str(fliud)
    
    def _rgb(self, im, flipud):
        rgbat = im.as_str(flipud)
        rgba = fromstring(rgbat[2], UInt8)
        rgba.shape = (rgbat[0], rgbat[1], 4)
        rgb = rgba[:,:,:3]
        return rgbat[0], rgbat[1], rgb.tostring()

    def _gray(self, im, flipud, rc=0.3, gc=0.59, bc=0.11):
        rgbat = im.as_str(flipud)
        rgba = fromstring(rgbat[2], UInt8)
        rgba.shape = (rgbat[0], rgbat[1], 4)
        rgba_f = rgba.astype(Float32)
        r = rgba_f[:,:,0]
        g = rgba_f[:,:,1]
        b = rgba_f[:,:,2]
        gray = (r*rc + g*gc + b*bc).astype(UInt8)
        return rgbat[0], rgbat[1], gray.tostring()

    def _hex_lines(self, s, chars_per_line=128):
        s = binascii.b2a_hex(s)
        nhex = len(s)
        lines = []
        for i in range(0,nhex,chars_per_line):
            limit = min(i+chars_per_line, nhex)
            lines.append(s[i:limit])
        return lines

    def draw_image(self, x, y, im, origin, bbox):
        """
        Draw the Image instance into the current axes; x is the
        distance in pixels from the left hand side of the canvas. y is
        the distance from the origin.  That is, if origin is upper, y
        is the distance from top.  If origin is lower, y is the
        distance from bottom

        origin is 'upper' or 'lower'

        bbox is a matplotlib.transforms.BBox instance for clipping, or
        None
        """

        flipud = origin=='lower'
        if im.is_grayscale:
            h, w, bits = self._gray(im, flipud)
            imagecmd = "image"
        else:
            h, w, bits = self._rgb(im, flipud)
            imagecmd = "false 3 colorimage"
        hexlines = '\n'.join(self._hex_lines(bits))

        xscale, yscale = w, h

        figh = self.height*72
        #print 'values', origin, flipud, figh, h, y
        if bbox is not None:
            clipx,clipy,clipw,cliph = bbox.get_bounds()
            clip = '%s clipbox' % _nums_to_str(clipw, cliph, clipx, clipy)
        if not flipud: y = figh-(y+h)
        ps = """gsave
%(clip)s
%(x)s %(y)s translate
%(xscale)s %(yscale)s scale
/DataString %(w)s string def
%(w)s %(h)s 8 [ %(w)s 0 0 -%(h)s 0 %(h)s ]
{
currentfile DataString readhexstring pop
} bind %(imagecmd)s
%(hexlines)s
grestore
""" % locals()
        self._pswriter.write(ps)
    
    def draw_line(self, gc, x0, y0, x1, y1):
        """
        Draw a single line from x0,y0 to x1,y1
        """
        ps = '%1.3f %1.3f m %1.3f %1.3f l'%(x0, y0, x1, y1)
        self._draw_ps(ps, gc, None, "line")

    def _draw_markers(self, gc, path, x, y, transform):
        """
        Draw the markers defined by path at each of the positions in x
        and y.  path coordinates are points, x and y coords will be
        transformed by the transform
        """
        if debugPS:
            self._pswriter.write("% markers\n")

        if transform.need_nonlinear():
            x,y = transform.nonlinear_only_numerix(x, y)

        # the a,b,c,d,tx,ty affine which transforms x and y
        vec6 = transform.as_vec6_val()
        # this defines a single vertex.  We need to define this as ps
        # function, properly stroked and filled with linewidth etc,
        # and then simply iterate over the x and y and call this
        # function at each position.  Eg, this is the path that is
        # relative to each x and y offset.
        ps = []
        for p in path:
            code = p[0]
            if code==MOVETO:
                mx, my = p[1:]
                ps.append('%1.3f %1.3f m')
            elif code==LINETO:
                mx, my = p[1:]
                ps.append('%1.3f %1.3f l')
            elif code==ENDPOLY:
                fill = p[1]
                if fill:  # we can get the fill color here
                    rgba = p[2:]
                    
        vertfunc = 'some magic ps function that draws the marker relative to an x,y point'
        # the gc contains the stroke width and color as always
        for i in xrange(len(x)):
            # for large numbers of markers you may need to chunk the
            # output, eg dump the ps in 1000 marker batches
            thisx = x[i]
            thisy = y[i]
            # apply affine transform x and y to define marker center
            #draw_marker_here

        print 'I did nothing!'

    def _draw_lines(self, gc, points):
        """
        Draw many lines.  'points' is a list of point coordinates.
        """
        # inline this for performance
        ps = ["%1.3f %1.3f m" % points[0]] 
        ps.extend(["%1.3f %1.3f l" % point for point in points[1:] ])
        self._draw_ps("\n".join(ps), gc, None)

    def draw_lines(self, gc, x, y):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        if debugPS:
            self._pswriter.write("% lines\n")
        start  = 0
        end    = 1000
        points = zip(x,y)
        
        while 1:
            to_draw = points[start:end]
            if not to_draw:
                break
            self._draw_lines(gc,to_draw)
            start = end
            end   += 1000

    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        # TODO: is there a better way to draw points in postscript?
        #       (use a small circle?)
        self.draw_line(gc, x, y, x+1, y+1)

    def draw_polygon(self, gc, rgbFace, points):
        """
        Draw a polygon.  points is a len vertices tuple, each element
        giving the x,y coords a vertex

        If rgbFace is not None, fill the poly with it.  gc
        is a GraphicsContext instance
        """
        ps = ["%s m\n" % _nums_to_str(*points[0])]
        ps.extend([ "%s l\n" % _nums_to_str(x, y) for x,y in points[1:] ])
        ps.append("closepath")
        self._draw_ps(''.join(ps), gc, rgbFace, "polygon")
        
    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        """
        Draw a rectangle with lower left at x,y with width and height.

        If gcFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """
        # TODO: use rectstroke
        ps = '%s box' % _nums_to_str(width, height, x, y)
        self._draw_ps(ps, gc, rgbFace, "rectangle")

    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        """
        draw a Text instance
        """
        # local to avoid repeated attribute lookups
        write = self._pswriter.write
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        if debugPS:
            write("% text\n")
        
        font = self._get_font(prop)
        font.set_text(s,0)

        self.set_color(*gc.get_rgb())
        self.set_font(font.get_sfnt()[(1,0,0,6)], prop.get_size_in_points())
        write("%s m\n"%_nums_to_str(x,y))
        if angle:
            write("gsave\n")
            write("%s rotate\n"%_num_to_str(angle))
        descent = font.get_descent() / 64.0
        if descent:
            write("0 %s rmoveto\n"%_num_to_str(descent))
        write("(%s) show\n"%quote_ps_string(s))
        if angle:
            write("grestore\n")

    def new_gc(self):
        return GraphicsContextPS()

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """
        Draw the math text using matplotlib.mathtext
        """
        if debugPS:
            self._pswriter.write("% mathtext\n")

        fontsize = prop.get_size_in_points()
        width, height, pswriter = math_parse_s_ps(s, 72, fontsize)
        self.set_color(*gc.get_rgb())
        thetext = pswriter.getvalue()
        ps = """gsave
%(x)f %(y)f translate
%(angle)f rotate
%(thetext)s
grestore
""" % locals()
        self._pswriter.write(ps)

    def _draw_ps(self, ps, gc, rgbFace, command=None):
        """
        Emit the PostScript sniplet 'ps' with all the attributes from 'gc'
        applied.  'ps' must consist of PostScript commands to construct a path.
        """
        # local variable eliminates all repeated attribute lookups
        write = self._pswriter.write
        
        if debugPS and command:
            write("% "+command+"\n")

        cliprect = gc.get_clip_rectangle()
        self.set_color(*gc.get_rgb())
        self.set_linewidth(gc.get_linewidth())
        # TODO: move the lookup into GraphicsContextPS
        jint = {'miter':0, 'round':1, 'bevel':2}[gc.get_joinstyle()]
        self.set_linejoin(jint)
        # TODO: move the lookup into GraphicsContextPS
        cint = {'butt':0, 'round':1, 'projecting':2}[gc.get_capstyle()]
        self.set_linecap(cint)
        self.set_linedash(*gc.get_dashes())
        if cliprect:
            x,y,w,h=cliprect
            write('gsave\n%1.3f %1.3f %1.3f %1.3f clipbox\n' % (w,h,x,y))
        # Jochen, is the strip necessary? - this could be a honking big string
        write(ps.strip())  
        write("\n")        
        if rgbFace:
            #print 'rgbface', rgbFace
            write("gsave\n")
            self.set_color(store=0, *rgbFace)
            write("fill\ngrestore\n")

        write("stroke\n")
        if cliprect:
            write("grestore\n")


class GraphicsContextPS(GraphicsContextBase):

    def set_linestyle(self, style):
        """
        Set the linestyle to be one of ('solid', 'dashed', 'dashdot',
        'dotted').
        """
        GraphicsContextBase.set_linestyle(self, style)
        offset, dashes = self.dashd[style]
        self.set_dashes(offset, dashes)

def new_figure_manager(num, *args, **kwargs):
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasPS(thisFig)
    manager = FigureManagerPS(canvas, num)
    return manager

def encodeTTFasPS(fontfile):
    """
    Encode a TrueType font file for embedding in a PS file.
    """
    fontfile = str(fontfile) # TODO: handle unicode filenames
    font = file(fontfile, 'rb')
    hexdata, data = [], font.read(65520)
    b2a_hex = binascii.b2a_hex
    while data:
        hexdata.append('<%s>\n' %
                       '\n'.join([b2a_hex(data[j:j+36]).upper()
                                  for j in range(0, len(data), 36)]) )
        data  = font.read(65520)
        
    hexdata = ''.join(hexdata)[:-2] + '00>'
    font    = FT2Font(fontfile)
    
    headtab  = font.get_sfnt_table('head')
    version  = '%d.%d' % headtab['version']
    revision = '%d.%d' % headtab['fontRevision']

    dictsize = 8
    fontname = font.postscript_name
    encoding = 'StandardEncoding'
    fontbbox = '[%d %d %d %d]' % font.bbox
    
    posttab  = font.get_sfnt_table('post')
    minmemory= posttab['minMemType42']
    maxmemory= posttab['maxMemType42']

    infosize = 7
    sfnt     = font.get_sfnt()
    notice   = sfnt[(1,0,0,0)]
    family   = sfnt[(1,0,0,1)]
    fullname = sfnt[(1,0,0,4)]
    iversion = sfnt[(1,0,0,5)]
    fixpitch = str(bool(posttab['isFixedPitch'])).lower()
    ulinepos = posttab['underlinePosition']
    ulinethk = posttab['underlineThickness']
    italicang= '(%d.%d)' % posttab['italicAngle']

    numglyphs = font.num_glyphs
    glyphs = []
    for j in range(numglyphs):
        glyphs.append('/%s %d def' % (font.get_glyph_name(j), j))
        if j != 0 and j%4 == 0:
            glyphs.append('\n')
        else:
            glyphs.append(' ')
    glyphs = ''.join(glyphs)
    data = ['%%!PS-TrueType-%(version)s-%(revision)s\n' % locals()]
    if maxmemory:
        data.append('%%%%VMusage: %(minmemory)d %(maxmemory)d' % locals())
    data.append("""%(dictsize)d dict begin
/FontName /%(fontname)s def
/FontMatrix [1 0 0 1 0 0] def
/FontType 42 def
/Encoding %(encoding)s def
/FontBBox %(fontbbox)s def
/PaintType 0 def
/FontInfo %(infosize)d dict dup begin
/Notice (%(notice)s) def
/FamilyName (%(family)s) def
/FullName (%(fullname)s) def
/version (%(iversion)s) def
/isFixedPitch %(fixpitch)s def
/UnderlinePosition %(ulinepos)s def
/UnderlineThickness %(ulinethk)s def
end readonly def
/sfnts [
%(hexdata)s
] def
/CharStrings %(numglyphs)d dict dup begin
%(glyphs)s
end readonly def
FontName currentdict end definefont pop""" % locals())
    return ''.join(data)


class FigureCanvasPS(FigureCanvasBase):
    basepath = get_data_path()

    def draw(self):
        pass
    
    def print_figure(self, outfile, dpi=72,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):
        """
        Render the figure to hardcopy.  Set the figure patch face and
        edge colors.  This is useful because some of the GUIs have a
        gray figure face color background and you'll probably want to
        override this on hardcopy

        If outfile is a string, it is interpreted as a file name.
        If the extension matches .ep* write encapsulated postscript,
        otherwise write a stand-alone PostScript file.

        If outfile is a file object, a stand-alone PostScript file is
        written into this file object.
        """

        if  isinstance(outfile, file):
            # assume plain PostScript and write to fileobject
            isEPSF = False
            fh = outfile
            needsClose = False
            title = None
        else:
            basename, ext = os.path.splitext(outfile)
            if not ext: outfile += '.ps'
            isEPSF = ext.lower().startswith('.ep')
            fh = file(outfile, 'w')
            needsClose = True
            title = outfile
        
        # center the figure on the paper
        self.figure.dpi.set(72)        # ignore the passsed dpi setting for PS
        width, height = self.figure.get_size_inches()

        if orientation=='landscape':
            isLandscape = True
            paperHeight, paperWidth = defaultPaperSize
        else:
            isLandscape = False
            paperWidth, paperHeight = defaultPaperSize

        xo = 72*0.5*(paperWidth - width)
        yo = 72*0.5*(paperHeight - height)

        l, b, w, h = self.figure.bbox.get_bounds()
        llx = xo
        lly = yo
        urx = llx + w
        ury = lly + h

        if isLandscape:
            xo, yo = 72*paperHeight - yo, xo
            llx, lly, urx, ury = lly, llx, ury, urx
            rotation = 90
        else:
            rotation = 0

        # generate PostScript code for the figure and store it in a string
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        self._pswriter = StringIO()
        renderer = RendererPS(width, height, self._pswriter)
        self.figure.draw(renderer)

        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)

        # write the PostScript headers
        if isEPSF:
            print >>fh, "%!PS-Adobe-3.0 EPSF-3.0"
        else:
            print >>fh, "%!PS-Adobe-3.0"
        if title: print >>fh, "%%Title: "+title
        print >>fh, ("%%Creator: matplotlib version "
                     +__version__+", http://matplotlib.sourceforge.net/")
        print >>fh, "%%CreationDate: "+time.ctime(time.time())
        if not isEPSF:
            if paperWidth > paperHeight:
                ostr="Landscape"
            else:
                ostr="Portrait"
            print >>fh, "%%Orientation: "+ostr
        print >>fh, "%%%%BoundingBox: %d %d %d %d" % (llx, lly, urx, ury)
        if not isEPSF: print >>fh, "%%Pages: 1"
        print >>fh, "%%EndComments"
        
        type42 = _type42 + [os.path.join(self.basepath, name) + '.ttf' \
                            for name in bakoma_fonts]
        print >>fh, "%%BeginProlog"
        print >>fh, "/mpldict %d dict def"%(len(_psDefs)+len(type42))
        print >>fh, "mpldict begin"
        for d in _psDefs:
            d=d.strip()
            for l in d.split('\n'):
                print >>fh, l.strip()
        for font in type42:
            font = str(font)  # TODO: handle unicode filenames
            print >>fh, "%%BeginFont: "+FT2Font(font).postscript_name
            print >>fh, encodeTTFasPS(font)
            print >>fh, "%%EndFont"
        print >>fh, "%%EndProlog"
        
        if not isEPSF: print >>fh, "%%Page: 1 1"
        print >>fh, "mpldict begin"
        print >>fh, "gsave"
        print >>fh, "%s translate"%_nums_to_str(xo, yo)
        if rotation:
            print >>fh, "%d rotate"%rotation
        print >>fh, "%s clipbox"%_nums_to_str(width*72, height*72, 0, 0)

        # write the figure
        print >>fh, self._pswriter.getvalue()

        # write the trailer
        print >>fh, "grestore"
        print >>fh, "end"
        print >>fh, "showpage"

        if not isEPSF: print >>fh, "%%EOF"
        if needsClose: fh.close()

class FigureManagerPS(FigureManagerBase):
    pass


FigureManager = FigureManagerPS


# The following Python dictionary _psDefs contains the entries for the
# PostScript dictionary mpldict.  This dictionary implements most of
# the matplotlib primitives and some abbreviations.
#
# References:
# http://www.adobe.com/products/postscript/pdfs/PLRM.pdf
# http://www.mactech.com/articles/mactech/Vol.09/09.04/PostscriptTutorial/
# http://www.math.ubc.ca/people/faculty/cass/graphics/text/www/
#
# Some comments about the implementation:
#
# Drawing ellipses:
#
# ellipse adds a counter-clockwise segment of an elliptical arc to the
# current path. The ellipse procedure takes six operands: the x and y
# coordinates of the center of the ellipse (the center is defined as
# the point of intersection of the major and minor axes), the
# ``radius'' of the ellipse in the x direction, the ``radius'' of the
# ellipse in the y direction, the starting angle of the elliptical arc
# and the ending angle of the elliptical arc.
#
# The basic strategy used in drawing the ellipse is to translate to
# the center of the ellipse, scale the user coordinate system by the x
# and y radius values, and then add a circular arc, centered at the
# origin with a 1 unit radius to the current path. We will be
# transforming the user coordinate system with the translate and
# rotate operators to add the elliptical arc segment but we don't want
# these transformations to affect other parts of the program. In other
# words, we would like to localize the effect of the transformations.
# Usually the gsave and grestore operators would be ideal candidates
# for this task.  Unfortunately gsave and grestore are inappropriate
# for this situation because we cannot save the arc segment that we
# have added to the path. Instead we will localize the effect of the
# transformations by saving the current transformation matrix and
# restoring it explicitly after we have added the elliptical arc to
# the path.

# The usage comments use the notation of the operator summary
# in the PostScript Language reference manual.
_psDefs = [
    # x y  *m*  -
    "/m { moveto } bind def",
    # x y  *l*  -
    "/l { lineto } bind def",
    # x y  *r*  -
    "/r { rlineto } bind def",
    # w h x y  *box*  -
    """/box {
      m
      1 index 0 r
      0 exch r
      neg 0 r
      closepath
    } bind def""",
    # w h x y  *clipbox*  -
    """/clipbox {
      box
      clip
      newpath
    } bind def""",
    # angle1 angle2 rx ry x y  *ellipse*  -
    """/ellipse {
      matrix currentmatrix 7 1 roll
      translate
      scale
      0 0 1 5 3 roll arc
      setmatrix
    } bind def"""
]
