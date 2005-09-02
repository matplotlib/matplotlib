"""
A PostScript backend, which can produce both PostScript .ps and

"""

from __future__ import division
import glob, math, md5, os, shutil, sys, time
def _fn_name(): return sys._getframe(1).f_code.co_name
    
from cStringIO import StringIO
from matplotlib import verbose, __version__, rcParams, get_data_path
from matplotlib._pylab_helpers import Gcf
import matplotlib.agg as agg
from matplotlib.afm import AFM
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase

from matplotlib.cbook import is_string_like, izip, reverse_dict
from matplotlib.figure import Figure

from matplotlib.font_manager import fontManager
from matplotlib.ft2font import FT2Font, KERNING_UNFITTED, KERNING_DEFAULT, KERNING_UNSCALED
from matplotlib.mathtext import math_parse_s_ps, bakoma_fonts
from matplotlib.text import Text
from matplotlib.texmanager import TexManager

from matplotlib.transforms import get_vec6_scales

from matplotlib.numerix import fromstring, UInt8, Float32, equal, alltrue, \
     nonzero, take, where, ones, put
import binascii
import re

backend_version = 'Level II'

debugPS = 1

papersize = {'executive': (7.5,11),
             'letter': (8.5,11),
             'legal': (8.5,14), 
             'ledger': (11,17),
             'a0': (33.11,46.81),
             'a1': (23.39,33.11),
             'a2': (16.54,23.39),
             'a3': (11.69,16.54),
             'a4': (8.27,11.69),
             'a5': (5.83,8.27),
             'a6': (4.13,5.83),
             'a7': (2.91,4.13),
             'a8': (2.07,2.91),
             'a9': (1.457,2.05),
             'a10': (1.02,1.457),
             'b0': (40.55,57.32),
             'b1': (28.66,40.55),
             'b2': (20.27,28.66),
             'b3': (14.33,20.27),
             'b4': (10.11,14.33),
             'b5': (7.16,10.11),
             'b6': (5.04,7.16),
             'c0': (36.10,51.06),
             'c1': (25.51,36.10),
             'c2': (18.03,25.51),
             'c3': (12.75,18.03),
             'c4': (9.01,12.75),
             'c5': (6.38,9.01),
             'c6': (4.49,6.38)}
defaultPaperType = rcParams['ps.papersize']
defaultPaperSize = papersize[defaultPaperType]

def _get_papersize(w,h):
    keys = papersize.keys()
    keys.sort()
    keys.reverse()
    for key in keys:
        if key.startswith('l'): continue
        val = papersize[key]
        # will the image fit within latex margins?
        if w+2 < val[0] and h+2 < val[1]: return val
    else: return papersize['a0']

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
_afmfontd = {}
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
        if rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
            self.texmanager = TexManager()

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
            if store: self.color = (r,g,b)

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
        if rcParams['ps.useafm']: return
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
        if rcParams['text.usetex']:
            fontsize = prop.get_size_in_points()
            l,b,r,t = self.texmanager.get_ps_bbox(s)
            w = (r-l)*fontsize/10.
            h = (t-b)*fontsize/10.
            #print s, w, h
            return w, h

        if rcParams['ps.useafm']:
            if ismath: s = s[1:-1]
            font = self._get_font_afm(prop)
            l,b,w,h = font.get_str_bbox(s)

            fontsize = prop.get_size_in_points()
            w *= 0.001*fontsize
            h *= 0.001*fontsize
            return w, h

            
        if ismath:
            width, height, pswriter = math_parse_s_ps(
                s, 72, prop.get_size_in_points())
            return width, height

        font = self._get_font_ttf(prop)
        font.set_text(s, 0.0)
        w, h = font.get_width_height()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        #print s, w, h
        return w, h

    def flipy(self):
        'return true if small y numbers are top for renderer'
        return False

    def _get_font_afm(self, prop):
        key = hash(prop)
        font = _afmfontd.get(key)
        if font is None:
            font = AFM(file(fontManager.findfont(prop, fontext='afm')))
            _afmfontd[key] = font
        return font

    def _get_font_ttf(self, prop):
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

    def _rgba(self, im):
        return im.as_rgba_str()
    
    def _rgb(self, im):
        h,w,s = im.as_rgba_str()
        
        rgba = fromstring(s, UInt8)
        rgba.shape = (h, w, 4)
        rgb = rgba[:,:,:3]
        return h, w, rgb.tostring()

    def _gray(self, im, rc=0.3, gc=0.59, bc=0.11):
        rgbat = im.as_rgba_str()
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

    def draw_image(self, x, y, im, bbox):
        """
        Draw the Image instance into the current axes; x is the
        distance in pixels from the left hand side of the canvas and y
        is the distance from bottom

        bbox is a matplotlib.transforms.BBox instance for clipping, or
        None
        """

        im.flipud_out()

        if im.is_grayscale:
            h, w, bits = self._gray(im)
            imagecmd = "image"
        else:
            h, w, bits = self._rgb(im)
            imagecmd = "false 3 colorimage"
        hexlines = '\n'.join(self._hex_lines(bits))

        xscale, yscale = w, h

        figh = self.height*72
        #print 'values', origin, flipud, figh, h, y

        if bbox is not None:
            clipx,clipy,clipw,cliph = bbox.get_bounds()
            clip = '%s clipbox' % _nums_to_str(clipw, cliph, clipx, clipy)
        #y = figh-(y+h)
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

        # unflip
        im.flipud_out()

    def draw_line(self, gc, x0, y0, x1, y1):
        """
        Draw a single line from x0,y0 to x1,y1
        """
        ps = '%1.3f %1.3f m %1.3f %1.3f l'%(x0, y0, x1, y1)
        self._draw_ps(ps, gc, None, "line")
        
    def _draw_markers(self, gc, path, rgbFace, x, y, transform):
        """
        Draw the markers defined by path at each of the positions in x
        and y.  path coordinates are points, x and y coords will be
        transformed by the transform
        """
        if debugPS: self._pswriter.write('% draw_markers \n')
        
        return 
        if rgbFace:
            if rgbFace[0]==rgbFace[0] and rgbFace[0]==rgbFace[2]:
                ps_color = '%1.3f setgray' % rgbFace[0]
            else:
                ps_color = '%1.3f %1.3f %1.3f setrgbcolor' % rgbFace

        #if transform.need_nonlinear():
        #    x,y,mask = transform.nonlinear_only_numerix(x, y, returnMask=1)
        #else:
        #    mask = ones(x.shape)
            
        x, y = transform.numerix_x_y(x, y)

        # the a,b,c,d,tx,ty affine which transforms x and y
        #vec6 = transform.as_vec6_val()
        #theta = (180 / pi) * math.atan2 (vec6[1], src[0])
        # this defines a single vertex.  We need to define this as ps
        # function, properly stroked and filled with linewidth etc,
        # and then simply iterate over the x and y and call this
        # function at each position.  Eg, this is the path that is
        # relative to each x and y offset.
        
        # construct the generic marker command:
        ps_cmd = ['gsave']
        ps_cmd.append('newpath')
        ps_cmd.append('translate')
        while 1:
            code, xp, yp = path.vertex()
            if code == agg.path_cmd_stop:
                ps_cmd.append('closepath') # Hack, path_cmd_end_poly not found
                break
            elif code == agg.path_cmd_move_to:
                ps_cmd.append('%1.3f %1.3f m' % (xp,yp))
            elif code == agg.path_cmd_line_to:
                ps_cmd.append('%1.3f %1.3f l' % (xp,yp))
            elif code == agg.path_cmd_curve3:
                pass
            elif code == agg.path_cmd_curve4:
                pass
            elif code == agg.path_cmd_end_poly:
                pass
                ps_cmd.append('closepath')
            elif code == agg.path_cmd_mask:
                pass
            else:
                pass
                #print code
                
        if rgbFace:
            ps_cmd.append('gsave')
            ps_cmd.append(ps_color)
            ps_cmd.append('fill')
            ps_cmd.append('grestore')
        ps_cmd.append('stroke')
        ps_cmd.append('grestore') # undo translate()
        ps_cmd = '\n'.join(ps_cmd)
        
        #self._pswriter.write(' '.join(['/marker {', ps_cmd, '} bind def\n']))
        #self._pswriter.write('[%s]' % ';'.join([float(val) for val in vec6]))        
        # Now evaluate the marker command at each marker location:
        start  = 0
        end    = 1000
        while start < len(x):

            to_draw = izip(x[start:end],y[start:end])
            ps = ['%1.3f %1.3f marker' % point for point in to_draw] 
            self._draw_ps("\n".join(ps), gc, None)
            start = end
            end   += 1000
            
    def draw_path(self,gc,rgbFace,path,trans):
        pass

    def _draw_lines(self, gc, points):
        """
        Draw many lines.  'points' is a list of point coordinates.
        """
        # inline this for performance
        ps = ["%1.3f %1.3f m" % points[0]] 
        ps.extend(["%1.3f %1.3f l" % point for point in points[1:] ])
        self._draw_ps("\n".join(ps), gc, None)


    def draw_lines(self, gc, x, y, transform=None):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        if debugPS: self._pswriter.write('% draw_lines \n')
 
        if transform:  # this won't be called if draw_markers is hidden
            #if transform.need_nonlinear():
            #    x, y = transform.nonlinear_only_numerix(x, y)
            x, y = transform.numerix_x_y(x, y)
        
        start  = 0
        end    = 1000
        points = zip(x,y)
        while start < len(x):
            to_draw = izip(x[start:end],y[start:end])
            ps = ["%1.3f %1.3f m" % to_draw.next()] 
            ps.extend(["%1.3f %1.3f l" % point for point in to_draw])
            self._draw_ps("\n".join(ps), gc, None)
            start = end
            end   += 1000
        
    def __draw_lines_hide(self, gc, x, y, transform=None):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        if debugPS: self._pswriter.write('% draw_lines \n')
    
    
        if transform:
            if transform.need_nonlinear():
                x, y, mask = transform.nonlinear_only_numerix(x, y, returnMask=1)
            else:
                mask = ones(x.shape)

        vec6 = transform.as_vec6_val()
        a,b,c,d,tx,ty = vec6
        sx, sy = get_vec6_scales(vec6)

        start  = 0
        end    = 1000
        points = zip(x,y)

        write = self._pswriter.write
        write('gsave\n')
        self.push_gc(gc)
        write('[%f %f %f %f %f %f] concat\n'%(a,b,c,d,tx,ty))                
        
        while start < len(x):
            # put moveto on all the bad data and on the first good
            # point after the bad data
            codes = where(mask[start:end+1], 'l', 'm')
            ind = nonzero(mask[start:end+1]==0)+1
            if ind[-1]>=len(codes):
                ind = ind[:-1]
            put(codes, ind, 'm')
            
            thisx = x[start:end+1]
            thisy = y[start:end+1]
            to_draw = izip(thisx, thisy, codes)
            if not to_draw:
                break

            ps = ['%1.3f %1.3f m' % to_draw.next()[:2]]
            ps.extend(["%1.3f %1.3f %c" % tup for tup in to_draw])
            # we don't want to scale the line width, etc so invert the
            # scale for the stroke
            ps.append('\ngsave %f %f scale stroke grestore\n'%(1./sx,1./sy))
            write('\n'.join(ps))
            start = end
            end   += 1000
        write("grestore\n")
        
    def draw_lines_old(self, gc, x, y):
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

    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!'):
        """
        draw a Text instance
        """
        w, h = self.get_text_width_height(s, prop, ismath)
        fontsize = prop.get_size_in_points()
        corr = 0#w/2*(fontsize-10)/10
        pos = _nums_to_str(x-corr, y)
        thetext = 'psmarker%d' % self.textcnt
        scale = float(fontsize/10.0)
        color = '%1.3f,%1.3f,%1.3f'% gc.get_rgb()
        tex = r'\color[rgb]{%s} %s' % (color, s)
        self.psfrag.append(r'\psfrag{%s}[bl][bl][%f][%f]{%s}'%(thetext, scale, angle, tex))
        ps = """\
gsave
%(pos)s moveto
(%(thetext)s)
show
grestore
    """ % locals()

        self._pswriter.write(ps)
        self.textcnt += 1

    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        """
        draw a Text instance
        """
        # local to avoid repeated attribute lookups

        
        write = self._pswriter.write
        if debugPS:
            write("% text\n")

        if rcParams['ps.useafm']:
            if ismath: s = s[1:-1]
            font = self._get_font_afm(prop)

            l,b,w,h = font.get_str_bbox(s)

            fontsize = prop.get_size_in_points()
            l *= 0.001*fontsize
            b *= 0.001*fontsize
            w *= 0.001*fontsize
            h *= 0.001*fontsize

            if angle==90: l,b = -b, l # todo generalize for arb rotations

            pos = _nums_to_str(x-l, y-b)
            thetext = '(%s)' % s
            fontname = font.get_fontname()
            fontsize = prop.get_size_in_points()
            rotate = '%1.1f rotate' % angle
            setcolor = '%1.3f %1.3f %1.3f setrgbcolor' % gc.get_rgb()
            #h = 0
            ps = """\
gsave
/%(fontname)s findfont
%(fontsize)s scalefont
setfont
%(pos)s moveto
%(rotate)s
%(thetext)s
%(setcolor)s
show
grestore
    """ % locals()
            self._draw_ps(ps, gc, None)

        elif ismath=='TeX':
            return self.tex(gc, x, y, s, prop, angle)

        elif ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        elif isinstance(s, unicode):
            return self.draw_unicode(gc, x, y, s, prop, angle)
        else:
            font = self._get_font_ttf(prop)
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

    def draw_unicode(self, gc, x, y, s, prop, angle):
        """draw a unicode string.  ps doesn't have unicode support, so
        we have to do this the hard way
        """


        font = self._get_font_ttf(prop)        

        self.set_color(*gc.get_rgb())
        self.set_font(font.get_sfnt()[(1,0,0,6)], prop.get_size_in_points())

        cmap = font.get_charmap()
        glyphd = reverse_dict(cmap)
        lastgind = None
        #print 'text', s
        lines = []
        thisx, thisy = 0,0
        for c in s:
            ccode = ord(c)
            gind = glyphd.get(ccode)
            if gind is None:
                ccode = ord('?')
                name = '.notdef'
                gind = 0
            else:
                name = font.get_glyph_name(gind)
            glyph = font.load_char(ccode)

            if lastgind is not None:
                kern = font.get_kerning(lastgind, gind, KERNING_UNFITTED)
            else:
                kern = 0
            lastgind = gind
            thisx += kern/64.0

            lines.append('%f %f m /%s glyphshow'%(thisx, thisy, name))
            thisx += glyph.linearHoriAdvance/65536.0             


        thetext = '\n'.join(lines)
        ps = """gsave
%(x)f %(y)f translate
%(angle)f rotate
%(thetext)s
grestore
""" % locals()
        self._pswriter.write(ps)
                

    def draw_mathtext(self, gc,
        x, y, s, prop, angle):
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

    def push_gc(self, gc):
        """
        Push the current onto stack
        """
        # local variable eliminates all repeated attribute lookups
        write = self._pswriter.write
        
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
            write('%1.3f %1.3f %1.3f %1.3f clipbox\n' % (w,h,x,y))
        write("\n")        


class GraphicsContextPS(GraphicsContextBase):
    pass


def new_figure_manager(num, *args, **kwargs):
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasPS(thisFig)
    manager = FigureManagerPS(canvas, num)
    return manager

def encodeTTFasPS(fontfile):
    """
    Encode a TrueType font file for embedding in a PS file.
    """
    font = file(fontfile, 'rb')
    hexdata, data = [], font.read(65520)
    b2a_hex = binascii.b2a_hex
    while data:
        hexdata.append('<%s>\n' %
                       '\n'.join([b2a_hex(data[j:j+36]).upper()
                                  for j in range(0, len(data), 36)]) )
        data  = font.read(65520)
        
    hexdata = ''.join(hexdata)[:-2] + '00>'
    font    = FT2Font(str(fontfile))
    
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
        
        If text.usetex is True in rc, a temporary pair of tex/eps files 
        are created to allow tex to handle the text. The final output 
        is a simple ps or eps file.
        """

        if  isinstance(outfile, file):
            # assume plain PostScript and write to fileobject
            isEPSF = False
            fh = outfile
            needsClose = False
            title = None
        else:
            basename, ext = os.path.splitext(outfile)
            if not ext: 
                if rcParams['text.usetex']: 
                    ext = '.eps'
                    outfile += ext
                else:
                    ext = '.ps'
                    outfile += ext
            if rcParams['text.usetex']:
                # need to make some temporary files so latex can run without
                # writing over something important.
                m = md5.md5(outfile)
                tmpname = m.hexdigest()
                
                epsfile = tmpname + '.eps'
                psfile = tmpname + '.ps'
                texfile = tmpname + '.tex'
                dvifile = tmpname + '.dvi'
                latexh = file(texfile, 'w')
                fh = file(epsfile, 'w')
            else:
                fh = file(outfile, 'w')
            
            isEPSF = ext.lower().startswith('.ep') or rcParams['text.usetex']
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
            print >>fh, "%%DocumentPaperSizes: "+defaultPaperType
        print >>fh, "%%%%BoundingBox: %d %d %d %d" % (llx, lly, urx, ury)
        if not isEPSF: print >>fh, "%%Pages: 1"
        print >>fh, "%%EndComments"
        
        Ndict = len(psDefs)
        print >>fh, "%%BeginProlog"
        if not rcParams['text.usetex']:
            type42 = _type42 + [os.path.join(self.basepath, name) + '.ttf' \
                                for name in bakoma_fonts]
            if not rcParams['ps.useafm']:
                Ndict += len(type42)
                
        print >>fh, "/mpldict %d dict def"%Ndict
        print >>fh, "mpldict begin"

        for d in psDefs:
            d=d.strip()
            for l in d.split('\n'):
                print >>fh, l.strip()
        if not rcParams['text.usetex']:
            if not rcParams['ps.useafm']:
                for font in type42:
                    print >>fh, "%%BeginFont: "+FT2Font(str(font)).postscript_name
                    print >>fh, encodeTTFasPS(font)
                    print >>fh, "%%EndFont"

        print >>fh, "%%EndProlog"
        
        if not isEPSF: print >>fh, "%%Page: 1 1"
        print >>fh, "mpldict begin"
        #print >>fh, "gsave"
        print >>fh, "%s translate"%_nums_to_str(xo, yo)
        if rotation:
            print >>fh, "%d rotate"%rotation
        print >>fh, "%s clipbox"%_nums_to_str(width*72, height*72, 0, 0)

        # write the figure
        print >>fh, self._pswriter.getvalue()

        # write the trailer
        #print >>fh, "grestore"
        print >>fh, "end"
        print >>fh, "showpage"

        if not isEPSF: print >>fh, "%%EOF"
        if needsClose: fh.close()
            
        if rcParams['text.usetex']:
            if rcParams['text.tex.engine'] == 'latex': 
                fontpackage = rcParams['font.latex.package']
            else: 
                fontpackage = 'type1cm'
            pw, ph = defaultPaperSize
            if width>pw-2 or height>ph-2: pw,ph = _get_papersize(width,height)
            print >>latexh, r"""\documentclass{scrartcl}
\usepackage{%s}
\usepackage{psfrag}
\usepackage[dvips]{graphicx}
\usepackage{color}
\pagestyle{empty}
\setlength{\paperheight}{%fin}
\setlength{\paperwidth}{%fin}
\setlength{\textwidth}{%fin}
\setlength{\textheight}{%fin}
\special{papersize=%fin,%fin}
\begin{document}
\begin{figure}[th!]
\begin{center}
%s
\includegraphics{%s}
\end{center}
\end{figure}
\end{document}
"""% (fontpackage, pw, ph, pw-2, ph-2, pw, ph, '\n'.join(renderer.psfrag), epsfile)

            latexh.close()

            command = 'latex -interaction=nonstopmode "%s"' % texfile
            verbose.report(command, 'debug-annoying')
            stdin, stdout, stderr = os.popen3(command)
            verbose.report(stdout.read(), 'debug-annoying')
            verbose.report(stderr.read(), 'helpful')
            command = 'dvips -R -T %fin,%fin -o "%s" "%s"' % (pw, ph, psfile, dvifile)
            verbose.report(command, 'debug-annoying')
            stdin, stdout, stderr = os.popen3(command)
            verbose.report(stdout.read(), 'debug-annoying')
            verbose.report(stderr.read(), 'helpful')
            os.remove(epsfile)
            if ext.startswith('.ep'):
                dpi = rcParams['ps.distiller.res']
                command = 'gs -dBATCH -dNOPAUSE -dSAFER -r%d \
                    -sDEVICE=epswrite -dLanguageLevel=2 -dEPSFitPage \
                    -sOutputFile="%s" "%s"'% (dpi, epsfile, psfile)
                verbose.report(command, 'debug-annoying')
                stdin, stdout, stderr = os.popen3(command)
                verbose.report(stdout.read(), 'debug-annoying')
                verbose.report(stderr.read(), 'helpful')
                shutil.move(epsfile, outfile)
            else: shutil.move(psfile, outfile)

            for fname in glob.glob(tmpname+'.*'):
                os.remove(fname)
                
        if rcParams['ps.usedistiller']:
            dpi = rcParams['ps.distiller.res']
            m = md5.md5(outfile)
            tmpfile = m.hexdigest()
            if ext.startswith('ep'):
                command = 'eps2eps -dSAFER -r%d "%s" "%s"'% (dpi, outfile, tmpfile)
            else:
                command = 'ps2ps -dSAFER -r%d "%s" "%s"'% (dpi, outfile, tmpfile)
            verbose.report(command, 'debug-annoying')
            stdin, stdout, stderr = os.popen3(command)
            verbose.report(stdout.read(), 'debug-annoying')
            verbose.report(stderr.read(), 'helpful')
            shutil.move(tmpfile, outfile)

class FigureManagerPS(FigureManagerBase):
    pass


FigureManager = FigureManagerPS


# The following Python dictionary psDefs contains the entries for the
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
psDefs = [
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
      newpath
      matrix currentmatrix 7 1 roll
      translate
      scale
      0 0 1 5 3 roll arc
      setmatrix
      closepath
    } bind def"""
]
