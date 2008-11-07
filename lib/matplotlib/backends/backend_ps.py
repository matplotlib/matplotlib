"""
A PostScript backend, which can produce both PostScript .ps and .eps
"""

from __future__ import division
import glob, math, os, shutil, sys, time
def _fn_name(): return sys._getframe(1).f_code.co_name

try:
    from hashlib import md5
except ImportError:
    from md5 import md5 #Deprecated in 2.5

from tempfile import gettempdir
from cStringIO import StringIO
from matplotlib import verbose, __version__, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.afm import AFM
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase

from matplotlib.cbook import is_string_like, get_realpath_and_stat, \
    is_writable_file_like, maxdict
from matplotlib.mlab import quad2cubic
from matplotlib.figure import Figure

from matplotlib.font_manager import findfont, is_opentype_cff_font
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT, LOAD_NO_HINTING
from matplotlib.ttconv import convert_ttf_to_ps
from matplotlib.mathtext import MathTextParser
from matplotlib._mathtext_data import uni2type1
from matplotlib.text import Text
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

import numpy as npy
import binascii
import re
try:
    set
except NameError:
    from sets import Set as set

if sys.platform.startswith('win'): cmd_split = '&'
else: cmd_split = ';'

backend_version = 'Level II'

debugPS = 0

papersize = {'letter': (8.5,11),
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
             'b7': (3.58,5.04),
             'b8': (2.51,3.58),
             'b9': (1.76,2.51),
             'b10': (1.26,1.76)}

def _get_papertype(w, h):
    keys = papersize.keys()
    keys.sort()
    keys.reverse()
    for key in keys:
        if key.startswith('l'): continue
        pw, ph = papersize[key]
        if (w < pw) and (h < ph): return key
    else:
        return 'a0'

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
    s=s.replace("'", "\\251")
    s=s.replace("`", "\\301")
    s=re.sub(r"[^ -~\n]", lambda x: r"\%03o"%ord(x.group()), s)
    return s


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
    return npy.alltrue(npy.equal(seq1, seq2))


class RendererPS(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    fontd = maxdict(50)
    afmfontd = maxdict(50)

    def __init__(self, width, height, pswriter, imagedpi=72):
        """
        Although postscript itself is dpi independent, we need to
        imform the image code about a requested dpi to generate high
        res images and them scale them before embeddin them
        """
        RendererBase.__init__(self)
        self.width = width
        self.height = height
        self._pswriter = pswriter
        if rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
        self.imagedpi = imagedpi
        if rcParams['path.simplify']:
            self.simplify = (width * imagedpi, height * imagedpi)
        else:
            self.simplify = None

        # current renderer state (None=uninitialised)
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None
        self.hatch = None
        self.image_magnification = imagedpi/72.0
        self._clip_paths = {}
        self._path_collection_id = 0

        self.used_characters = {}
        self.mathtext_parser = MathTextParser("PS")

    def track_characters(self, font, s):
        """Keeps track of which characters are required from
        each font."""
        realpath, stat_key = get_realpath_and_stat(font.fname)
        used_characters = self.used_characters.setdefault(
            stat_key, (realpath, set()))
        used_characters[1].update([ord(x) for x in s])

    def merge_used_characters(self, other):
        for stat_key, (realpath, charset) in other.items():
            used_characters = self.used_characters.setdefault(
                stat_key, (realpath, set()))
            used_characters[1].update(charset)

    def set_color(self, r, g, b, store=1):
        if (r,g,b) != self.color:
            if r==g and r==b:
                self._pswriter.write("%1.3f setgray\n"%r)
            else:
                self._pswriter.write("%1.3f %1.3f %1.3f setrgbcolor\n"%(r,g,b))
            if store: self.color = (r,g,b)

    def set_linewidth(self, linewidth, store=1):
        if linewidth != self.linewidth:
            self._pswriter.write("%1.3f setlinewidth\n"%linewidth)
            if store: self.linewidth = linewidth

    def set_linejoin(self, linejoin, store=1):
        if linejoin != self.linejoin:
            self._pswriter.write("%d setlinejoin\n"%linejoin)
            if store: self.linejoin = linejoin

    def set_linecap(self, linecap, store=1):
        if linecap != self.linecap:
            self._pswriter.write("%d setlinecap\n"%linecap)
            if store: self.linecap = linecap

    def set_linedash(self, offset, seq, store=1):
        if self.linedash is not None:
            oldo, oldseq = self.linedash
            if seq_allequal(seq, oldseq): return

        if seq is not None and len(seq):
            s="[%s] %d setdash\n"%(_nums_to_str(*seq), offset)
            self._pswriter.write(s)
        else:
            self._pswriter.write("[] 0 setdash\n")
        if store: self.linedash = (offset,seq)

    def set_font(self, fontname, fontsize, store=1):
        if rcParams['ps.useafm']: return
        if (fontname,fontsize) != (self.fontname,self.fontsize):
            out = ("/%s findfont\n"
                   "%1.3f scalefont\n"
                   "setfont\n" % (fontname,fontsize))

            self._pswriter.write(out)
            if store: self.fontname = fontname
            if store: self.fontsize = fontsize

    def set_hatch(self, hatch):
        """
        hatch can be one of:
            /   - diagonal hatching
            \   - back diagonal
            |   - vertical
            -   - horizontal
            +   - crossed
            X   - crossed diagonal

        letters can be combined, in which case all the specified
        hatchings are done

        if same letter repeats, it increases the density of hatching
        in that direction
        """
        hatches = {'horiz':0, 'vert':0, 'diag1':0, 'diag2':0}

        for letter in hatch:
            if   (letter == '/'):    hatches['diag2'] += 1
            elif (letter == '\\'):   hatches['diag1'] += 1
            elif (letter == '|'):    hatches['vert']  += 1
            elif (letter == '-'):    hatches['horiz'] += 1
            elif (letter == '+'):
                hatches['horiz'] += 1
                hatches['vert'] += 1
            elif (letter.lower() == 'x'):
                hatches['diag1'] += 1
                hatches['diag2'] += 1

        def do_hatch(angle, density):
            if (density == 0): return ""
            return """\
  gsave
   eoclip %s rotate 0.0 0.0 0.0 0.0 setrgbcolor 0 setlinewidth
   /hatchgap %d def
   pathbbox /hatchb exch def /hatchr exch def /hatcht exch def /hatchl exch def
   hatchl cvi hatchgap idiv hatchgap mul
   hatchgap
   hatchr cvi hatchgap idiv hatchgap mul
   {hatcht m 0 hatchb hatcht sub r }
   for
   stroke
  grestore
 """ % (angle, 12/density)
        self._pswriter.write("gsave\n")
        self._pswriter.write(do_hatch(90, hatches['horiz']))
        self._pswriter.write(do_hatch(0, hatches['vert']))
        self._pswriter.write(do_hatch(45, hatches['diag1']))
        self._pswriter.write(do_hatch(-45, hatches['diag2']))
        self._pswriter.write("grestore\n")

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height

    def get_text_width_height_descent(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop

        """
        if rcParams['text.usetex']:
            texmanager = self.get_texmanager()
            fontsize = prop.get_size_in_points()
            l,b,r,t = texmanager.get_ps_bbox(s, fontsize)
            w = (r-l)
            h = (t-b)
            # TODO: We need a way to get a good baseline from
            # text.usetex
            return w, h, 0

        if ismath:
            width, height, descent, pswriter, used_characters = \
                self.mathtext_parser.parse(s, 72, prop)
            return width, height, descent

        if rcParams['ps.useafm']:
            if ismath: s = s[1:-1]
            font = self._get_font_afm(prop)
            l,b,w,h,d = font.get_str_bbox_and_descent(s)

            fontsize = prop.get_size_in_points()
            scale = 0.001*fontsize
            w *= scale
            h *= scale
            d *= scale
            return w, h, d

        font = self._get_font_ttf(prop)
        font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
        w, h = font.get_width_height()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d = font.get_descent()
        d /= 64.0
        #print s, w, h
        return w, h, d

    def flipy(self):
        'return true if small y numbers are top for renderer'
        return False

    def _get_font_afm(self, prop):
        key = hash(prop)
        font = self.afmfontd.get(key)
        if font is None:
            fname = findfont(prop, fontext='afm')
            font = self.afmfontd.get(fname)
            if font is None:
                font = AFM(file(findfont(prop, fontext='afm')))
                self.afmfontd[fname] = font
            self.afmfontd[key] = font
        return font

    def _get_font_ttf(self, prop):
        key = hash(prop)
        font = self.fontd.get(key)
        if font is None:
            fname = findfont(prop)
            font = self.fontd.get(fname)
            if font is None:
                font = FT2Font(str(fname))
                self.fontd[fname] = font
            self.fontd[key] = font
        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, 72.0)
        return font

    def _rgba(self, im):
        return im.as_rgba_str()

    def _rgb(self, im):
        h,w,s = im.as_rgba_str()

        rgba = npy.fromstring(s, npy.uint8)
        rgba.shape = (h, w, 4)
        rgb = rgba[:,:,:3]
        return h, w, rgb.tostring()

    def _gray(self, im, rc=0.3, gc=0.59, bc=0.11):
        rgbat = im.as_rgba_str()
        rgba = npy.fromstring(rgbat[2], npy.uint8)
        rgba.shape = (rgbat[0], rgbat[1], 4)
        rgba_f = rgba.astype(npy.float32)
        r = rgba_f[:,:,0]
        g = rgba_f[:,:,1]
        b = rgba_f[:,:,2]
        gray = (r*rc + g*gc + b*bc).astype(npy.uint8)
        return rgbat[0], rgbat[1], gray.tostring()

    def _hex_lines(self, s, chars_per_line=128):
        s = binascii.b2a_hex(s)
        nhex = len(s)
        lines = []
        for i in range(0,nhex,chars_per_line):
            limit = min(i+chars_per_line, nhex)
            lines.append(s[i:limit])
        return lines

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to draw_image.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return self.image_magnification

    def draw_image(self, x, y, im, bbox, clippath=None, clippath_trans=None):
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

        xscale, yscale = (
            w/self.image_magnification, h/self.image_magnification)

        figh = self.height*72
        #print 'values', origin, flipud, figh, h, y

        clip = []
        if bbox is not None:
            clipx,clipy,clipw,cliph = bbox.bounds
            clip.append('%s clipbox' % _nums_to_str(clipw, cliph, clipx, clipy))
        if clippath is not None:
            id = self._get_clip_path(clippath, clippath_trans)
            clip.append('%s' % id)
        clip = '\n'.join(clip)

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

    def _convert_path(self, path, transform, simplify=None):
        path = transform.transform_path(path)

        ps = []
        last_points = None
        for points, code in path.iter_segments(simplify):
            if code == Path.MOVETO:
                ps.append("%g %g m" % tuple(points))
            elif code == Path.LINETO:
                ps.append("%g %g l" % tuple(points))
            elif code == Path.CURVE3:
                points = quad2cubic(*(list(last_points[-2:]) + list(points)))
                ps.append("%g %g %g %g %g %g c" %
                          tuple(points[2:]))
            elif code == Path.CURVE4:
                ps.append("%g %g %g %g %g %g c" % tuple(points))
            elif code == Path.CLOSEPOLY:
                ps.append("cl")
            last_points = points

        ps = "\n".join(ps)
        return ps

    def _get_clip_path(self, clippath, clippath_transform):
        id = self._clip_paths.get((clippath, clippath_transform))
        if id is None:
            id = 'c%x' % len(self._clip_paths)
            ps_cmd = ['/%s {' % id]
            ps_cmd.append(self._convert_path(clippath, clippath_transform))
            ps_cmd.extend(['clip', 'newpath', '} bind def\n'])
            self._pswriter.write('\n'.join(ps_cmd))
            self._clip_paths[(clippath, clippath_transform)] = id
        return id

    def draw_path(self, gc, path, transform, rgbFace=None):
        """
        Draws a Path instance using the given affine transform.
        """
        ps = self._convert_path(path, transform, self.simplify)
        self._draw_ps(ps, gc, rgbFace)

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        """
        Draw the markers defined by path at each of the positions in x
        and y.  path coordinates are points, x and y coords will be
        transformed by the transform
        """
        if debugPS: self._pswriter.write('% draw_markers \n')

        write = self._pswriter.write

        if rgbFace:
            if rgbFace[0]==rgbFace[1] and rgbFace[0]==rgbFace[2]:
                ps_color = '%1.3f setgray' % rgbFace[0]
            else:
                ps_color = '%1.3f %1.3f %1.3f setrgbcolor' % rgbFace

        # construct the generic marker command:
        ps_cmd = ['/o {', 'gsave', 'newpath', 'translate'] # dont want the translate to be global
        ps_cmd.append(self._convert_path(marker_path, marker_trans))

        if rgbFace:
            ps_cmd.extend(['gsave', ps_color, 'fill', 'grestore'])

        ps_cmd.extend(['stroke', 'grestore', '} bind def'])

        tpath = trans.transform_path(path)
        for vertices, code in tpath.iter_segments():
            if len(vertices):
                x, y = vertices[-2:]
                ps_cmd.append("%g %g o" % (x, y))

        ps = '\n'.join(ps_cmd)
        self._draw_ps(ps, gc, rgbFace, fill=False, stroke=False)

    def draw_path_collection(self, master_transform, cliprect, clippath,
                             clippath_trans, paths, all_transforms, offsets,
                             offsetTrans, facecolors, edgecolors, linewidths,
                             linestyles, antialiaseds, urls):
        write = self._pswriter.write

        path_codes = []
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(
            master_transform, paths, all_transforms)):
            name = 'p%x_%x' % (self._path_collection_id, i)
            ps_cmd = ['/%s {' % name,
                      'newpath', 'translate']
            ps_cmd.append(self._convert_path(path, transform))
            ps_cmd.extend(['} bind def\n'])
            write('\n'.join(ps_cmd))
            path_codes.append(name)

        for xo, yo, path_id, gc, rgbFace in self._iter_collection(
            path_codes, cliprect, clippath, clippath_trans,
            offsets, offsetTrans, facecolors, edgecolors,
            linewidths, linestyles, antialiaseds, urls):

            ps = "%g %g %s" % (xo, yo, path_id)
            self._draw_ps(ps, gc, rgbFace)

        self._path_collection_id += 1

    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!'):
        """
        draw a Text instance
        """
        w, h, bl = self.get_text_width_height_descent(s, prop, ismath)
        fontsize = prop.get_size_in_points()
        corr = 0#w/2*(fontsize-10)/10
        pos = _nums_to_str(x-corr, y)
        thetext = 'psmarker%d' % self.textcnt
        color = '%1.3f,%1.3f,%1.3f'% gc.get_rgb()[:3]
        fontcmd = {'sans-serif' : r'{\sffamily %s}',
               'monospace'  : r'{\ttfamily %s}'}.get(
                rcParams['font.family'], r'{\rmfamily %s}')
        s = fontcmd % s
        tex = r'\color[rgb]{%s} %s' % (color, s)
        self.psfrag.append(r'\psfrag{%s}[bl][bl][1][%f]{\fontsize{%f}{%f}%s}'%(thetext, angle, fontsize, fontsize*1.25, tex))
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

        if ismath=='TeX':
            return self.tex(gc, x, y, s, prop, angle)

        elif ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        elif isinstance(s, unicode):
            return self.draw_unicode(gc, x, y, s, prop, angle)

        elif rcParams['ps.useafm']:
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
            setcolor = '%1.3f %1.3f %1.3f setrgbcolor' % gc.get_rgb()[:3]
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

        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0, flags=LOAD_NO_HINTING)
            self.track_characters(font, s)

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
        if rcParams['ps.useafm']:
            self.set_color(*gc.get_rgb())

            font = self._get_font_afm(prop)
            fontname = font.get_fontname()
            fontsize = prop.get_size_in_points()
            scale = 0.001*fontsize

            thisx = 0
            thisy = font.get_str_bbox_and_descent(s)[4] * scale
            last_name = None
            lines = []
            for c in s:
                name = uni2type1.get(ord(c), 'question')
                try:
                    width = font.get_width_from_char_name(name)
                except KeyError:
                    name = 'question'
                    width = font.get_width_char('?')
                if last_name is not None:
                    kern = font.get_kern_dist_from_name(last_name, name)
                else:
                    kern = 0
                last_name = name
                thisx += kern * scale

                lines.append('%f %f m /%s glyphshow'%(thisx, thisy, name))

                thisx += width * scale

            thetext = "\n".join(lines)
            ps = """\
gsave
/%(fontname)s findfont
%(fontsize)s scalefont
setfont
%(x)f %(y)f translate
%(angle)f rotate
%(thetext)s
grestore
    """ % locals()
            self._pswriter.write(ps)

        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0, flags=LOAD_NO_HINTING)
            self.track_characters(font, s)

            self.set_color(*gc.get_rgb())
            self.set_font(font.get_sfnt()[(1,0,0,6)], prop.get_size_in_points())

            cmap = font.get_charmap()
            lastgind = None
            #print 'text', s
            lines = []
            thisx = 0
            thisy = font.get_descent() / 64.0
            for c in s:
                ccode = ord(c)
                gind = cmap.get(ccode)
                if gind is None:
                    ccode = ord('?')
                    name = '.notdef'
                    gind = 0
                else:
                    name = font.get_glyph_name(gind)
                glyph = font.load_char(ccode, flags=LOAD_NO_HINTING)

                if lastgind is not None:
                    kern = font.get_kerning(lastgind, gind, KERNING_DEFAULT)
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

        width, height, descent, pswriter, used_characters = \
            self.mathtext_parser.parse(s, 72, prop)
        self.merge_used_characters(used_characters)
        self.set_color(*gc.get_rgb())
        thetext = pswriter.getvalue()
        ps = """gsave
%(x)f %(y)f translate
%(angle)f rotate
%(thetext)s
grestore
""" % locals()
        self._pswriter.write(ps)

    def _draw_ps(self, ps, gc, rgbFace, fill=True, stroke=True, command=None):
        """
        Emit the PostScript sniplet 'ps' with all the attributes from 'gc'
        applied.  'ps' must consist of PostScript commands to construct a path.

        The fill and/or stroke kwargs can be set to False if the
        'ps' string already includes filling and/or stroking, in
        which case _draw_ps is just supplying properties and
        clipping.
        """
        # local variable eliminates all repeated attribute lookups
        write = self._pswriter.write
        if debugPS and command:
            write("% "+command+"\n")
        mightstroke = (gc.get_linewidth() > 0.0 and
                  (len(gc.get_rgb()) <= 3 or gc.get_rgb()[3] != 0.0))
        stroke = stroke and mightstroke
        fill = (fill and rgbFace is not None and
                (len(rgbFace) <= 3 or rgbFace[3] != 0.0))

        if mightstroke:
            self.set_linewidth(gc.get_linewidth())
            jint = gc.get_joinstyle()
            self.set_linejoin(jint)
            cint = gc.get_capstyle()
            self.set_linecap(cint)
            self.set_linedash(*gc.get_dashes())
            self.set_color(*gc.get_rgb()[:3])
        write('gsave\n')

        cliprect = gc.get_clip_rectangle()
        if cliprect:
            x,y,w,h=cliprect.bounds
            write('%1.4g %1.4g %1.4g %1.4g clipbox\n' % (w,h,x,y))
        clippath, clippath_trans = gc.get_clip_path()
        if clippath:
            id = self._get_clip_path(clippath, clippath_trans)
            write('%s\n' % id)

        # Jochen, is the strip necessary? - this could be a honking big string
        write(ps.strip())
        write("\n")

        if fill:
            if stroke:
                write("gsave\n")
                self.set_color(store=0, *rgbFace[:3])
                write("fill\ngrestore\n")
            else:
                self.set_color(store=0, *rgbFace[:3])
                write("fill\n")

        hatch = gc.get_hatch()
        if hatch:
            self.set_hatch(hatch)

        if stroke:
            write("stroke\n")

        write("grestore\n")



class GraphicsContextPS(GraphicsContextBase):
    def get_capstyle(self):
        return {'butt':0,
                'round':1,
                'projecting':2}[GraphicsContextBase.get_capstyle(self)]

    def get_joinstyle(self):
        return {'miter':0,
                'round':1,
                'bevel':2}[GraphicsContextBase.get_joinstyle(self)]


def new_figure_manager(num, *args, **kwargs):
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    canvas = FigureCanvasPS(thisFig)
    manager = FigureManagerPS(canvas, num)
    return manager

class FigureCanvasPS(FigureCanvasBase):
    def draw(self):
        pass

    filetypes = {'ps'  : 'Postscript',
                 'eps' : 'Encapsulated Postscript'}

    def get_default_filetype(self):
        return 'ps'

    def print_ps(self, outfile, *args, **kwargs):
        return self._print_ps(outfile, 'ps', *args, **kwargs)

    def print_eps(self, outfile, *args, **kwargs):
        return self._print_ps(outfile, 'eps', *args, **kwargs)

    def _print_ps(self, outfile, format, *args, **kwargs):
        papertype = kwargs.get("papertype", rcParams['ps.papersize'])
        papertype = papertype.lower()
        if papertype == 'auto':
            pass
        elif papertype not in papersize:
            raise RuntimeError( '%s is not a valid papertype. Use one \
                    of %s'% (papertype, ', '.join( papersize.keys() )) )

        orientation = kwargs.get("orientation", "portrait").lower()
        if orientation == 'landscape': isLandscape = True
        elif orientation == 'portrait': isLandscape = False
        else: raise RuntimeError('Orientation must be "portrait" or "landscape"')

        self.figure.set_dpi(72) # Override the dpi kwarg
        imagedpi = kwargs.get("dpi", 72)
        facecolor = kwargs.get("facecolor", "w")
        edgecolor = kwargs.get("edgecolor", "w")

        if rcParams['text.usetex']:
            self._print_figure_tex(outfile, format, imagedpi, facecolor, edgecolor,
                                   orientation, isLandscape, papertype)
        else:
            self._print_figure(outfile, format, imagedpi, facecolor, edgecolor,
                               orientation, isLandscape, papertype)

    def _print_figure(self, outfile, format, dpi=72, facecolor='w', edgecolor='w',
                      orientation='portrait', isLandscape=False, papertype=None):
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
        isEPSF = format == 'eps'
        passed_in_file_object = False
        if is_string_like(outfile):
            title = outfile
            tmpfile = os.path.join(gettempdir(), md5(outfile).hexdigest())
        elif is_writable_file_like(outfile):
            title = None
            tmpfile = os.path.join(gettempdir(), md5(str(hash(outfile))).hexdigest())
            passed_in_file_object = True
        else:
            raise ValueError("outfile must be a path or a file-like object")
        fh = file(tmpfile, 'w')

        # find the appropriate papertype
        width, height = self.figure.get_size_inches()
        if papertype == 'auto':
            if isLandscape: papertype = _get_papertype(height, width)
            else: papertype = _get_papertype(width, height)

        if isLandscape: paperHeight, paperWidth = papersize[papertype]
        else: paperWidth, paperHeight = papersize[papertype]

        if rcParams['ps.usedistiller'] and not papertype == 'auto':
            # distillers will improperly clip eps files if the pagesize is
            # too small
            if width>paperWidth or height>paperHeight:
                if isLandscape:
                    papertype = _get_papertype(height, width)
                    paperHeight, paperWidth = papersize[papertype]
                else:
                    papertype = _get_papertype(width, height)
                    paperWidth, paperHeight = papersize[papertype]

        # center the figure on the paper
        xo = 72*0.5*(paperWidth - width)
        yo = 72*0.5*(paperHeight - height)

        l, b, w, h = self.figure.bbox.bounds
        llx = xo
        lly = yo
        urx = llx + w
        ury = lly + h
        rotation = 0
        if isLandscape:
            llx, lly, urx, ury = lly, llx, ury, urx
            xo, yo = 72*paperHeight - yo, xo
            rotation = 90
        bbox = (llx, lly, urx, ury)

        # generate PostScript code for the figure and store it in a string
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        self._pswriter = StringIO()
        renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        self.figure.draw(renderer)

        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)

        # write the PostScript headers
        if isEPSF: print >>fh, "%!PS-Adobe-3.0 EPSF-3.0"
        else: print >>fh, "%!PS-Adobe-3.0"
        if title: print >>fh, "%%Title: "+title
        print >>fh, ("%%Creator: matplotlib version "
                     +__version__+", http://matplotlib.sourceforge.net/")
        print >>fh, "%%CreationDate: "+time.ctime(time.time())
        print >>fh, "%%Orientation: " + orientation
        if not isEPSF: print >>fh, "%%DocumentPaperSizes: "+papertype
        print >>fh, "%%%%BoundingBox: %d %d %d %d" % bbox
        if not isEPSF: print >>fh, "%%Pages: 1"
        print >>fh, "%%EndComments"

        Ndict = len(psDefs)
        print >>fh, "%%BeginProlog"
        if not rcParams['ps.useafm']:
            Ndict += len(renderer.used_characters)
        print >>fh, "/mpldict %d dict def"%Ndict
        print >>fh, "mpldict begin"
        for d in psDefs:
            d=d.strip()
            for l in d.split('\n'):
                print >>fh, l.strip()
        if not rcParams['ps.useafm']:
            for font_filename, chars in renderer.used_characters.values():
                if len(chars):
                    font = FT2Font(font_filename)
                    cmap = font.get_charmap()
                    glyph_ids = []
                    for c in chars:
                        gind = cmap.get(c) or 0
                        glyph_ids.append(gind)
                    # The ttf to ps (subsetting) support doesn't work for
                    # OpenType fonts that are Postscript inside (like the
                    # STIX fonts).  This will simply turn that off to avoid
                    # errors.
                    if is_opentype_cff_font(font_filename):
                        raise RuntimeError("OpenType CFF fonts can not be saved using the internal Postscript backend at this time.\nConsider using the Cairo backend.")
                    else:
                        fonttype = rcParams['ps.fonttype']
                        convert_ttf_to_ps(font_filename, fh, rcParams['ps.fonttype'], glyph_ids)
        print >>fh, "end"
        print >>fh, "%%EndProlog"

        if not isEPSF: print >>fh, "%%Page: 1 1"
        print >>fh, "mpldict begin"
        #print >>fh, "gsave"
        print >>fh, "%s translate"%_nums_to_str(xo, yo)
        if rotation: print >>fh, "%d rotate"%rotation
        print >>fh, "%s clipbox"%_nums_to_str(width*72, height*72, 0, 0)

        # write the figure
        print >>fh, self._pswriter.getvalue()

        # write the trailer
        #print >>fh, "grestore"
        print >>fh, "end"
        print >>fh, "showpage"
        if not isEPSF: print >>fh, "%%EOF"
        fh.close()

        if rcParams['ps.usedistiller'] == 'ghostscript':
            gs_distill(tmpfile, isEPSF, ptype=papertype, bbox=bbox)
        elif rcParams['ps.usedistiller'] == 'xpdf':
            xpdf_distill(tmpfile, isEPSF, ptype=papertype, bbox=bbox)

        if passed_in_file_object:
            fh = file(tmpfile)
            print >>outfile, fh.read()
        else:
            shutil.move(tmpfile, outfile)

    def _print_figure_tex(self, outfile, format, dpi, facecolor, edgecolor,
                          orientation, isLandscape, papertype):
        """
        If text.usetex is True in rc, a temporary pair of tex/eps files
        are created to allow tex to manage the text layout via the PSFrags
        package. These files are processed to yield the final ps or eps file.
        """
        isEPSF = format == 'eps'
        title = outfile

        # write to a temp file, we'll move it to outfile when done
        tmpfile = os.path.join(gettempdir(), md5(outfile).hexdigest())
        fh = file(tmpfile, 'w')

        self.figure.dpi = 72 # ignore the dpi kwarg
        width, height = self.figure.get_size_inches()
        xo = 0
        yo = 0

        l, b, w, h = self.figure.bbox.bounds
        llx = xo
        lly = yo
        urx = llx + w
        ury = lly + h
        bbox = (llx, lly, urx, ury)

        # generate PostScript code for the figure and store it in a string
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        self._pswriter = StringIO()
        renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        self.figure.draw(renderer)

        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)

        # write the Encapsulated PostScript headers
        print >>fh, "%!PS-Adobe-3.0 EPSF-3.0"
        if title: print >>fh, "%%Title: "+title
        print >>fh, ("%%Creator: matplotlib version "
                     +__version__+", http://matplotlib.sourceforge.net/")
        print >>fh, "%%CreationDate: "+time.ctime(time.time())
        print >>fh, "%%%%BoundingBox: %d %d %d %d" % bbox
        print >>fh, "%%EndComments"

        Ndict = len(psDefs)
        print >>fh, "%%BeginProlog"
        print >>fh, "/mpldict %d dict def"%Ndict
        print >>fh, "mpldict begin"
        for d in psDefs:
            d=d.strip()
            for l in d.split('\n'):
                print >>fh, l.strip()
        print >>fh, "end"
        print >>fh, "%%EndProlog"

        print >>fh, "mpldict begin"
        #print >>fh, "gsave"
        print >>fh, "%s translate"%_nums_to_str(xo, yo)
        print >>fh, "%s clipbox"%_nums_to_str(width*72, height*72, 0, 0)

        # write the figure
        print >>fh, self._pswriter.getvalue()

        # write the trailer
        #print >>fh, "grestore"
        print >>fh, "end"
        print >>fh, "showpage"
        fh.close()

        if isLandscape: # now we are ready to rotate
            isLandscape = True
            width, height = height, width
            bbox = (lly, llx, ury, urx)
        temp_papertype = _get_papertype(width, height)
        if papertype=='auto':
            papertype = temp_papertype
            paperWidth, paperHeight = papersize[temp_papertype]
        else:
            paperWidth, paperHeight = papersize[papertype]
            if (width>paperWidth or height>paperHeight) and isEPSF:
                paperWidth, paperHeight = papersize[temp_papertype]
                verbose.report('Your figure is too big to fit on %s paper. %s \
paper will be used to prevent clipping.'%(papertype, temp_papertype), 'helpful')

        texmanager = renderer.get_texmanager()
        font_preamble = texmanager.get_font_preamble()
        custom_preamble = texmanager.get_custom_preamble()

        convert_psfrags(tmpfile, renderer.psfrag, font_preamble,
                        custom_preamble, paperWidth, paperHeight,
                        orientation)

        if rcParams['ps.usedistiller'] == 'ghostscript':
            gs_distill(tmpfile, isEPSF, ptype=papertype, bbox=bbox)
        elif rcParams['ps.usedistiller'] == 'xpdf':
            xpdf_distill(tmpfile, isEPSF, ptype=papertype, bbox=bbox)
        elif rcParams['text.usetex']:
            if False: pass # for debugging
            else: gs_distill(tmpfile, isEPSF, ptype=papertype, bbox=bbox)

        if  isinstance(outfile, file):
            fh = file(tmpfile)
            print >>outfile, fh.read()
        else: shutil.move(tmpfile, outfile)

def convert_psfrags(tmpfile, psfrags, font_preamble, custom_preamble,
                    paperWidth, paperHeight, orientation):
    """
    When we want to use the LaTeX backend with postscript, we write PSFrag tags
    to a temporary postscript file, each one marking a position for LaTeX to
    render some text. convert_psfrags generates a LaTeX document containing the
    commands to convert those tags to text. LaTeX/dvips produces the postscript
    file that includes the actual text.
    """
    tmpdir = os.path.split(tmpfile)[0]
    epsfile = tmpfile+'.eps'
    shutil.move(tmpfile, epsfile)
    latexfile = tmpfile+'.tex'
    outfile = tmpfile+'.output'
    latexh = file(latexfile, 'w')
    dvifile = tmpfile+'.dvi'
    psfile = tmpfile+'.ps'

    if orientation=='landscape': angle = 90
    else: angle = 0

    if rcParams['text.latex.unicode']:
        unicode_preamble = """\usepackage{ucs}
\usepackage[utf8x]{inputenc}"""
    else:
        unicode_preamble = ''

    s = r"""\documentclass{article}
%s
%s
%s
\usepackage[dvips, papersize={%sin,%sin}, body={%sin,%sin}, margin={0in,0in}]{geometry}
\usepackage{psfrag}
\usepackage[dvips]{graphicx}
\usepackage{color}
\pagestyle{empty}
\begin{document}
\begin{figure}
\centering
\leavevmode
%s
\includegraphics*[angle=%s]{%s}
\end{figure}
\end{document}
"""% (font_preamble, unicode_preamble, custom_preamble, paperWidth, paperHeight,
      paperWidth, paperHeight,
      '\n'.join(psfrags), angle, os.path.split(epsfile)[-1])

    if rcParams['text.latex.unicode']:
        latexh.write(s.encode('utf8'))
    else:
        try:
            latexh.write(s)
        except UnicodeEncodeError, err:
            verbose.report("You are using unicode and latex, but have "
                           "not enabled the matplotlib 'text.latex.unicode' "
                           "rcParam.", 'helpful')
            raise

    latexh.close()

    # the split drive part of the command is necessary for windows users with
    # multiple
    if sys.platform == 'win32': precmd = '%s &&'% os.path.splitdrive(tmpdir)[0]
    else: precmd = ''
    command = '%s cd "%s" && latex -interaction=nonstopmode "%s" > "%s"'\
                %(precmd, tmpdir, latexfile, outfile)
    verbose.report(command, 'debug')
    exit_status = os.system(command)
    fh = file(outfile)
    if exit_status:
        raise RuntimeError('LaTeX was not able to process your file:\
\nHere is the full report generated by LaTeX: \n\n%s'% fh.read())
    else: verbose.report(fh.read(), 'debug')
    fh.close()
    os.remove(outfile)

    command = '%s cd "%s" && dvips -q -R0 -o "%s" "%s" > "%s"'%(precmd, tmpdir,
                os.path.split(psfile)[-1], os.path.split(dvifile)[-1], outfile)
    verbose.report(command, 'debug')
    exit_status = os.system(command)
    fh = file(outfile)
    if exit_status: raise RuntimeError('dvips was not able to \
process the following file:\n%s\nHere is the full report generated by dvips: \
\n\n'% dvifile + fh.read())
    else: verbose.report(fh.read(), 'debug')
    fh.close()
    os.remove(outfile)
    os.remove(epsfile)
    shutil.move(psfile, tmpfile)
    if not debugPS:
        for fname in glob.glob(tmpfile+'.*'):
            os.remove(fname)


def gs_distill(tmpfile, eps=False, ptype='letter', bbox=None):
    """
    Use ghostscript's pswrite or epswrite device to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. The output is low-level, converting text to outlines.
    """
    paper = '-sPAPERSIZE=%s'% ptype
    psfile = tmpfile + '.ps'
    outfile = tmpfile + '.output'
    dpi = rcParams['ps.distiller.res']
    if sys.platform == 'win32': gs_exe = 'gswin32c'
    else: gs_exe = 'gs'
    command = '%s -dBATCH -dNOPAUSE -r%d -sDEVICE=pswrite %s -sOutputFile="%s" \
                "%s" > "%s"'% (gs_exe, dpi, paper, psfile, tmpfile, outfile)
    verbose.report(command, 'debug')
    exit_status = os.system(command)
    fh = file(outfile)
    if exit_status: raise RuntimeError('ghostscript was not able to process \
your image.\nHere is the full report generated by ghostscript:\n\n' + fh.read())
    else: verbose.report(fh.read(), 'debug')
    fh.close()
    os.remove(outfile)
    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)
    if eps:
        pstoeps(tmpfile, bbox)


def xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None):
    """
    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. This distiller is preferred, generating high-level postscript
    output that treats text as text.
    """
    pdffile = tmpfile + '.pdf'
    psfile = tmpfile + '.ps'
    outfile = tmpfile + '.output'
    command = 'ps2pdf -dAutoFilterColorImages=false \
-sColorImageFilter=FlateEncode -sPAPERSIZE=%s "%s" "%s" > "%s"'% \
(ptype, tmpfile, pdffile, outfile)
    if sys.platform == 'win32': command = command.replace('=', '#')
    verbose.report(command, 'debug')
    exit_status = os.system(command)
    fh = file(outfile)
    if exit_status: raise RuntimeError('ps2pdf was not able to process your \
image.\n\Here is the report generated by ghostscript:\n\n' + fh.read())
    else: verbose.report(fh.read(), 'debug')
    fh.close()
    os.remove(outfile)
    command = 'pdftops -paper match -level2 "%s" "%s" > "%s"'% \
                (pdffile, psfile, outfile)
    verbose.report(command, 'debug')
    exit_status = os.system(command)
    fh = file(outfile)
    if exit_status: raise RuntimeError('pdftops was not able to process your \
image.\nHere is the full report generated by pdftops: \n\n' + fh.read())
    else: verbose.report(fh.read(), 'debug')
    fh.close()
    os.remove(outfile)
    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)
    if eps:
        pstoeps(tmpfile, bbox)
    for fname in glob.glob(tmpfile+'.*'):
        os.remove(fname)


def get_bbox(tmpfile, bbox):
    """
    Use ghostscript's bbox device to find the center of the bounding box. Return
    an appropriately sized bbox centered around that point. A bit of a hack.
    """
    outfile = tmpfile + '.output'
    if sys.platform == 'win32': gs_exe = 'gswin32c'
    else: gs_exe = 'gs'
    command = '%s -dBATCH -dNOPAUSE -sDEVICE=bbox "%s"' %\
                (gs_exe, tmpfile)
    verbose.report(command, 'debug')
    stdin, stdout, stderr = os.popen3(command)
    verbose.report(stdout.read(), 'debug-annoying')
    bbox_info = stderr.read()
    verbose.report(bbox_info, 'helpful')
    bbox_found = re.search('%%HiResBoundingBox: .*', bbox_info)
    if bbox_found:
        bbox_info = bbox_found.group()
    else:
        raise RuntimeError('Ghostscript was not able to extract a bounding box.\
Here is the Ghostscript output:\n\n%s'% bbox_info)
    l, b, r, t = [float(i) for i in bbox_info.split()[-4:]]

    # this is a hack to deal with the fact that ghostscript does not return the
    # intended bbox, but a tight bbox. For now, we just center the ink in the
    # intended bbox. This is not ideal, users may intend the ink to not be
    # centered.
    if bbox is None:
        l, b, r, t = (l-1, b-1, r+1, t+1)
    else:
        x = (l+r)/2
        y = (b+t)/2
        dx = (bbox[2]-bbox[0])/2
        dy = (bbox[3]-bbox[1])/2
        l,b,r,t = (x-dx, y-dy, x+dx, y+dy)

    bbox_info = '%%%%BoundingBox: %d %d %d %d' % (l, b, npy.ceil(r), npy.ceil(t))
    hires_bbox_info = '%%%%HiResBoundingBox: %.6f %.6f %.6f %.6f' % (l, b, r, t)

    return '\n'.join([bbox_info, hires_bbox_info])


def pstoeps(tmpfile, bbox):
    """
    Convert the postscript to encapsulated postscript.
    """
    bbox_info = get_bbox(tmpfile, bbox)

    epsfile = tmpfile + '.eps'
    epsh = file(epsfile, 'w')

    tmph = file(tmpfile)
    line = tmph.readline()
    # Modify the header:
    while line:
        if line.startswith('%!PS'):
            print >>epsh, "%!PS-Adobe-3.0 EPSF-3.0"
            print >>epsh, bbox_info
        elif line.startswith('%%EndComments'):
            epsh.write(line)
            print >>epsh, '%%BeginProlog'
            print >>epsh, 'save'
            print >>epsh, 'countdictstack'
            print >>epsh, 'mark'
            print >>epsh, 'newpath'
            print >>epsh, '/showpage {} def'
            print >>epsh, '/setpagedevice {pop} def'
            print >>epsh, '%%EndProlog'
            print >>epsh, '%%Page 1 1'
            break
        elif line.startswith('%%Bound') \
            or line.startswith('%%HiResBound') \
            or line.startswith('%%Pages'):
            pass
        else:
            epsh.write(line)
        line = tmph.readline()
    # Now rewrite the rest of the file, and modify the trailer.
    # This is done in a second loop such that the header of the embedded
    # eps file is not modified.
    line = tmph.readline()
    while line:
        if line.startswith('%%Trailer'):
            print >>epsh, '%%Trailer'
            print >>epsh, 'cleartomark'
            print >>epsh, 'countdictstack'
            print >>epsh, 'exch sub { end } repeat'
            print >>epsh, 'restore'
            if rcParams['ps.usedistiller'] == 'xpdf':
                # remove extraneous "end" operator:
                line = tmph.readline()
        else:
            epsh.write(line)
        line = tmph.readline()

    tmph.close()
    epsh.close()
    os.remove(tmpfile)
    shutil.move(epsfile, tmpfile)


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

# The usage comments use the notation of the operator summary
# in the PostScript Language reference manual.
psDefs = [
    # x y  *m*  -
    "/m { moveto } bind def",
    # x y  *l*  -
    "/l { lineto } bind def",
    # x y  *r*  -
    "/r { rlineto } bind def",
    # x1 y1 x2 y2 x y *c*  -
    "/c { curveto } bind def",
    # *closepath*  -
    "/cl { closepath } bind def",
    # w h x y  *box*  -
    """/box {
      m
      1 index 0 r
      0 exch r
      neg 0 r
      cl
    } bind def""",
    # w h x y  *clipbox*  -
    """/clipbox {
      box
      clip
      newpath
    } bind def""",
]
