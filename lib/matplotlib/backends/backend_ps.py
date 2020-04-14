"""
A PostScript backend, which can produce both PostScript .ps and .eps.
"""

import datetime
from enum import Enum
import glob
from io import StringIO, TextIOWrapper
import logging
import os
import pathlib
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import textwrap
import time

import numpy as np

import matplotlib as mpl
from matplotlib import (
    cbook, _path, __version__, rcParams, checkdep_ghostscript)
from matplotlib import _text_layout
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
    RendererBase)
from matplotlib.cbook import (get_realpath_and_stat, is_writable_file_like,
                              file_requires_unicode)
from matplotlib.font_manager import is_opentype_cff_font, get_font
from matplotlib.ft2font import LOAD_NO_HINTING
from matplotlib.ttconv import convert_ttf_to_ps
from matplotlib.mathtext import MathTextParser
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps

_log = logging.getLogger(__name__)

backend_version = 'Level II'

debugPS = 0


class PsBackendHelper:

    def __init__(self):
        self._cached = {}

    @cbook.deprecated("3.1")
    @property
    def gs_exe(self):
        """
        executable name of ghostscript.
        """
        try:
            return self._cached["gs_exe"]
        except KeyError:
            pass

        gs_exe, gs_version = checkdep_ghostscript()
        if gs_exe is None:
            gs_exe = 'gs'

        self._cached["gs_exe"] = str(gs_exe)
        return str(gs_exe)

    @cbook.deprecated("3.1")
    @property
    def gs_version(self):
        """
        version of ghostscript.
        """
        try:
            return self._cached["gs_version"]
        except KeyError:
            pass

        s = subprocess.Popen(
            [self.gs_exe, "--version"], stdout=subprocess.PIPE)
        pipe, stderr = s.communicate()
        ver = pipe.decode('ascii')
        try:
            gs_version = tuple(map(int, ver.strip().split(".")))
        except ValueError:
            # if something went wrong parsing return null version number
            gs_version = (0, 0)
        self._cached["gs_version"] = gs_version
        return gs_version

    @cbook.deprecated("3.1")
    @property
    def supports_ps2write(self):
        """
        True if the installed ghostscript supports ps2write device.
        """
        return self.gs_version[0] >= 9


ps_backend_helper = PsBackendHelper()


papersize = {'letter': (8.5, 11),
             'legal': (8.5, 14),
             'ledger': (11, 17),
             'a0': (33.11, 46.81),
             'a1': (23.39, 33.11),
             'a2': (16.54, 23.39),
             'a3': (11.69, 16.54),
             'a4': (8.27, 11.69),
             'a5': (5.83, 8.27),
             'a6': (4.13, 5.83),
             'a7': (2.91, 4.13),
             'a8': (2.07, 2.91),
             'a9': (1.457, 2.05),
             'a10': (1.02, 1.457),
             'b0': (40.55, 57.32),
             'b1': (28.66, 40.55),
             'b2': (20.27, 28.66),
             'b3': (14.33, 20.27),
             'b4': (10.11, 14.33),
             'b5': (7.16, 10.11),
             'b6': (5.04, 7.16),
             'b7': (3.58, 5.04),
             'b8': (2.51, 3.58),
             'b9': (1.76, 2.51),
             'b10': (1.26, 1.76)}


def _get_papertype(w, h):
    for key, (pw, ph) in sorted(papersize.items(), reverse=True):
        if key.startswith('l'):
            continue
        if w < pw and h < ph:
            return key
    return 'a0'


def _num_to_str(val):
    if isinstance(val, str):
        return val

    ival = int(val)
    if val == ival:
        return str(ival)

    s = "%1.3f" % val
    s = s.rstrip("0")
    s = s.rstrip(".")
    return s


def _nums_to_str(*args):
    return ' '.join(map(_num_to_str, args))


def quote_ps_string(s):
    "Quote dangerous characters of S for use in a PostScript string constant."
    s = s.replace(b"\\", b"\\\\")
    s = s.replace(b"(", b"\\(")
    s = s.replace(b")", b"\\)")
    s = s.replace(b"'", b"\\251")
    s = s.replace(b"`", b"\\301")
    s = re.sub(br"[^ -~\n]", lambda x: br"\%03o" % ord(x.group()), s)
    return s.decode('ascii')


def _move_path_to_path_or_stream(src, dst):
    """
    Move the contents of file at *src* to path-or-filelike *dst*.

    If *dst* is a path, the metadata of *src* are *not* copied.
    """
    if is_writable_file_like(dst):
        fh = (open(src, 'r', encoding='latin-1')
              if file_requires_unicode(dst)
              else open(src, 'rb'))
        with fh:
            shutil.copyfileobj(fh, dst)
    else:
        shutil.move(src, dst, copy_function=shutil.copyfile)


class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    @property
    @cbook.deprecated("3.1")
    def afmfontd(self, _cache=cbook.maxdict(50)):
        return _cache

    _afm_font_dir = cbook._get_data_path("fonts/afm")
    _use_afm_rc_name = "ps.useafm"

    def __init__(self, width, height, pswriter, imagedpi=72):
        # Although postscript itself is dpi independent, we need to inform the
        # image code about a requested dpi to generate high resolution images
        # and them scale them before embedding them.
        RendererBase.__init__(self)
        self.width = width
        self.height = height
        self._pswriter = pswriter
        if rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
        self.imagedpi = imagedpi

        # current renderer state (None=uninitialised)
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None
        self._hatches = {}
        self.image_magnification = imagedpi / 72
        self._clip_paths = {}
        self._path_collection_id = 0

        self.used_characters = {}
        self.mathtext_parser = MathTextParser("PS")

    def track_characters(self, font, s):
        """Keeps track of which characters are required from each font."""
        realpath, stat_key = get_realpath_and_stat(font.fname)
        used_characters = self.used_characters.setdefault(
            stat_key, (realpath, set()))
        used_characters[1].update(map(ord, s))

    def merge_used_characters(self, other):
        for stat_key, (realpath, charset) in other.items():
            used_characters = self.used_characters.setdefault(
                stat_key, (realpath, set()))
            used_characters[1].update(charset)

    def set_color(self, r, g, b, store=1):
        if (r, g, b) != self.color:
            if r == g and r == b:
                self._pswriter.write("%1.3f setgray\n" % r)
            else:
                self._pswriter.write(
                    "%1.3f %1.3f %1.3f setrgbcolor\n" % (r, g, b))
            if store:
                self.color = (r, g, b)

    def set_linewidth(self, linewidth, store=1):
        linewidth = float(linewidth)
        if linewidth != self.linewidth:
            self._pswriter.write("%1.3f setlinewidth\n" % linewidth)
            if store:
                self.linewidth = linewidth

    def set_linejoin(self, linejoin, store=1):
        if linejoin != self.linejoin:
            self._pswriter.write("%d setlinejoin\n" % linejoin)
            if store:
                self.linejoin = linejoin

    def set_linecap(self, linecap, store=1):
        if linecap != self.linecap:
            self._pswriter.write("%d setlinecap\n" % linecap)
            if store:
                self.linecap = linecap

    def set_linedash(self, offset, seq, store=1):
        if self.linedash is not None:
            oldo, oldseq = self.linedash
            if np.array_equal(seq, oldseq) and oldo == offset:
                return

        if seq is not None and len(seq):
            s = "[%s] %d setdash\n" % (_nums_to_str(*seq), offset)
            self._pswriter.write(s)
        else:
            self._pswriter.write("[] 0 setdash\n")
        if store:
            self.linedash = (offset, seq)

    def set_font(self, fontname, fontsize, store=1):
        if rcParams['ps.useafm']:
            return
        if (fontname, fontsize) != (self.fontname, self.fontsize):
            out = ("/%s findfont\n"
                   "%1.3f scalefont\n"
                   "setfont\n" % (fontname, fontsize))
            self._pswriter.write(out)
            if store:
                self.fontname = fontname
                self.fontsize = fontsize

    def create_hatch(self, hatch):
        sidelen = 72
        if hatch in self._hatches:
            return self._hatches[hatch]
        name = 'H%d' % len(self._hatches)
        linewidth = rcParams['hatch.linewidth']
        pageheight = self.height * 72
        self._pswriter.write(f"""\
  << /PatternType 1
     /PaintType 2
     /TilingType 2
     /BBox[0 0 {sidelen:d} {sidelen:d}]
     /XStep {sidelen:d}
     /YStep {sidelen:d}

     /PaintProc {{
        pop
        {linewidth:f} setlinewidth
{self._convert_path(
    Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False)}
        gsave
        fill
        grestore
        stroke
     }} bind
   >>
   matrix
   0.0 {pageheight:f} translate
   makepattern
   /{name} exch def
""")
        self._hatches[hatch] = name
        return name

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to draw_image.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return self.image_magnification

    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited

        h, w = im.shape[:2]
        imagecmd = "false 3 colorimage"
        data = im[::-1, :, :3]  # Vertically flipped rgb values.
        # data.tobytes().hex() has no spaces, so can be linewrapped by relying
        # on textwrap.fill breaking long words.
        hexlines = textwrap.fill(data.tobytes().hex(), 128)

        if transform is None:
            matrix = "1 0 0 1 0 0"
            xscale = w / self.image_magnification
            yscale = h / self.image_magnification
        else:
            matrix = " ".join(map(str, transform.frozen().to_values()))
            xscale = 1.0
            yscale = 1.0

        figh = self.height * 72

        bbox = gc.get_clip_rectangle()
        clippath, clippath_trans = gc.get_clip_path()

        clip = []
        if bbox is not None:
            clipx, clipy, clipw, cliph = bbox.bounds
            clip.append(
                '%s clipbox' % _nums_to_str(clipw, cliph, clipx, clipy))
        if clippath is not None:
            id = self._get_clip_path(clippath, clippath_trans)
            clip.append('%s' % id)
        clip = '\n'.join(clip)

        self._pswriter.write(f"""\
gsave
{clip}
{x:f} {y:f} translate
[{matrix}] concat
{xscale:f} {yscale:f} scale
/DataString {w:d} string def
{w:d} {h:d} 8 [ {w:d} 0 0 -{h:d} 0 {h:d} ]
{{
currentfile DataString readhexstring pop
}} bind {imagecmd}
{hexlines}
grestore
""")

    def _convert_path(self, path, transform, clip=False, simplify=None):
        if clip:
            clip = (0.0, 0.0, self.width * 72.0, self.height * 72.0)
        else:
            clip = None
        return _path.convert_to_string(
            path, transform, clip, simplify, None,
            6, [b'm', b'l', b'', b'c', b'cl'], True).decode('ascii')

    def _get_clip_path(self, clippath, clippath_transform):
        key = (clippath, id(clippath_transform))
        pid = self._clip_paths.get(key)
        if pid is None:
            pid = 'c%x' % len(self._clip_paths)
            clippath_bytes = self._convert_path(
                clippath, clippath_transform, simplify=False)
            self._pswriter.write(f"""\
/{pid} {{
{clippath_bytes}
clip
newpath
}} bind def
""")
            self._clip_paths[key] = pid
        return pid

    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring inherited
        clip = rgbFace is None and gc.get_hatch_path() is None
        simplify = path.should_simplify and clip
        ps = self._convert_path(path, transform, clip=clip, simplify=simplify)
        self._draw_ps(ps, gc, rgbFace)

    def draw_markers(
            self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # docstring inherited

        if debugPS:
            self._pswriter.write('% draw_markers \n')

        ps_color = (
            None
            if _is_transparent(rgbFace)
            else '%1.3f setgray' % rgbFace[0]
            if rgbFace[0] == rgbFace[1] == rgbFace[2]
            else '%1.3f %1.3f %1.3f setrgbcolor' % rgbFace[:3])

        # construct the generic marker command:

        # don't want the translate to be global
        ps_cmd = ['/o {', 'gsave', 'newpath', 'translate']

        lw = gc.get_linewidth()
        alpha = (gc.get_alpha()
                 if gc.get_forced_alpha() or len(gc.get_rgb()) == 3
                 else gc.get_rgb()[3])
        stroke = lw > 0 and alpha > 0
        if stroke:
            ps_cmd.append('%.1f setlinewidth' % lw)
            jint = gc.get_joinstyle()
            ps_cmd.append('%d setlinejoin' % jint)
            cint = gc.get_capstyle()
            ps_cmd.append('%d setlinecap' % cint)

        ps_cmd.append(self._convert_path(marker_path, marker_trans,
                                         simplify=False))

        if rgbFace:
            if stroke:
                ps_cmd.append('gsave')
            if ps_color:
                ps_cmd.extend([ps_color, 'fill'])
            if stroke:
                ps_cmd.append('grestore')

        if stroke:
            ps_cmd.append('stroke')
        ps_cmd.extend(['grestore', '} bind def'])

        for vertices, code in path.iter_segments(
                trans,
                clip=(0, 0, self.width*72, self.height*72),
                simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                ps_cmd.append("%g %g o" % (x, y))

        ps = '\n'.join(ps_cmd)
        self._draw_ps(ps, gc, rgbFace, fill=False, stroke=False)

    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offsetTrans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        # Is the optimization worth it? Rough calculation:
        # cost of emitting a path in-line is
        #     (len_path + 2) * uses_per_path
        # cost of definition+use is
        #     (len_path + 3) + 3 * uses_per_path
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        uses_per_path = self._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        should_do_optimization = \
            len_path + 3 * uses_per_path + 3 < (len_path + 2) * uses_per_path
        if not should_do_optimization:
            return RendererBase.draw_path_collection(
                self, gc, master_transform, paths, all_transforms,
                offsets, offsetTrans, facecolors, edgecolors,
                linewidths, linestyles, antialiaseds, urls,
                offset_position)

        write = self._pswriter.write

        path_codes = []
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(
                master_transform, paths, all_transforms)):
            name = 'p%x_%x' % (self._path_collection_id, i)
            path_bytes = self._convert_path(path, transform, simplify=False)
            write(f"""\
/{name} {{
newpath
translate
{path_bytes}
}} bind def
""")
            path_codes.append(name)

        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(
                gc, master_transform, all_transforms, path_codes, offsets,
                offsetTrans, facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            ps = "%g %g %s" % (xo, yo, path_id)
            self._draw_ps(ps, gc0, rgbFace)

        self._path_collection_id += 1

    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!', mtext=None):
        # docstring inherited

        w, h, bl = self.get_text_width_height_descent(s, prop, ismath)
        fontsize = prop.get_size_in_points()
        thetext = 'psmarker%d' % self.textcnt
        color = '%1.3f,%1.3f,%1.3f' % gc.get_rgb()[:3]
        fontcmd = {'sans-serif': r'{\sffamily %s}',
                   'monospace': r'{\ttfamily %s}'}.get(
                       rcParams['font.family'][0], r'{\rmfamily %s}')
        s = fontcmd % s
        tex = r'\color[rgb]{%s} %s' % (color, s)

        corr = 0  # w/2*(fontsize-10)/10
        if rcParams['text.latex.preview']:
            # use baseline alignment!
            pos = _nums_to_str(x-corr, y)
            self.psfrag.append(
                r'\psfrag{%s}[Bl][Bl][1][%f]{\fontsize{%f}{%f}%s}' % (
                    thetext, angle, fontsize, fontsize*1.25, tex))
        else:
            # Stick to the bottom alignment.
            pos = _nums_to_str(x-corr, y-bl)
            self.psfrag.append(
                r'\psfrag{%s}[bl][bl][1][%f]{\fontsize{%f}{%f}%s}' % (
                    thetext, angle, fontsize, fontsize*1.25, tex))

        self._pswriter.write(f"""\
gsave
{pos} moveto
({thetext})
show
grestore
""")
        self.textcnt += 1

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        # local to avoid repeated attribute lookups
        write = self._pswriter.write
        if debugPS:
            write("% text\n")

        if _is_transparent(gc.get_rgb()):
            return  # Special handling for fully transparent.

        if ismath == 'TeX':
            return self.draw_tex(gc, x, y, s, prop, angle)

        elif ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        elif rcParams['ps.useafm']:
            self.set_color(*gc.get_rgb())

            font = self._get_font_afm(prop)
            fontname = font.get_fontname()
            fontsize = prop.get_size_in_points()
            scale = 0.001 * fontsize

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

                lines.append('%f %f m /%s glyphshow' % (thisx, thisy, name))

                thisx += width * scale

            thetext = "\n".join(lines)
            self._pswriter.write(f"""\
gsave
/{fontname} findfont
{fontsize} scalefont
setfont
{x:f} {y:f} translate
{angle:f} rotate
{thetext}
grestore
""")

        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0, flags=LOAD_NO_HINTING)
            self.track_characters(font, s)

            self.set_color(*gc.get_rgb())
            ps_name = (font.postscript_name
                       .encode('ascii', 'replace').decode('ascii'))
            self.set_font(ps_name, prop.get_size_in_points())

            thetext = '\n'.join(
                '%f 0 m /%s glyphshow' % (x, font.get_glyph_name(glyph_idx))
                for glyph_idx, x in _text_layout.layout(s, font))
            self._pswriter.write(f"""\
gsave
{x:f} {y:f} translate
{angle:f} rotate
{thetext}
grestore
""")

    def new_gc(self):
        # docstring inherited
        return GraphicsContextPS()

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw the math text using matplotlib.mathtext."""
        if debugPS:
            self._pswriter.write("% mathtext\n")

        width, height, descent, pswriter, used_characters = \
            self.mathtext_parser.parse(s, 72, prop)
        self.merge_used_characters(used_characters)
        self.set_color(*gc.get_rgb())
        thetext = pswriter.getvalue()
        self._pswriter.write(f"""\
gsave
{x:f} {y:f} translate
{angle:f} rotate
{thetext}
grestore
""")

    def draw_gouraud_triangle(self, gc, points, colors, trans):
        self.draw_gouraud_triangles(gc, points.reshape((1, 3, 2)),
                                    colors.reshape((1, 3, 4)), trans)

    def draw_gouraud_triangles(self, gc, points, colors, trans):
        assert len(points) == len(colors)
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] == 4

        shape = points.shape
        flat_points = points.reshape((shape[0] * shape[1], 2))
        flat_points = trans.transform(flat_points)
        flat_colors = colors.reshape((shape[0] * shape[1], 4))
        points_min = np.min(flat_points, axis=0) - (1 << 12)
        points_max = np.max(flat_points, axis=0) + (1 << 12)
        factor = np.ceil((2 ** 32 - 1) / (points_max - points_min))

        xmin, ymin = points_min
        xmax, ymax = points_max

        streamarr = np.empty(
            (shape[0] * shape[1],),
            dtype=[('flags', 'u1'),
                   ('points', '>u4', (2,)),
                   ('colors', 'u1', (3,))])
        streamarr['flags'] = 0
        streamarr['points'] = (flat_points - points_min) * factor
        streamarr['colors'] = flat_colors[:, :3] * 255.0

        stream = quote_ps_string(streamarr.tobytes())

        self._pswriter.write(f"""\
gsave
<< /ShadingType 4
   /ColorSpace [/DeviceRGB]
   /BitsPerCoordinate 32
   /BitsPerComponent 8
   /BitsPerFlag 8
   /AntiAlias true
   /Decode [ {xmin:f} {xmax:f} {ymin:f} {ymax:f} 0 1 0 1 0 1 ]
   /DataSource ({stream})
>>
shfill
grestore
""")

    def _draw_ps(self, ps, gc, rgbFace, fill=True, stroke=True, command=None):
        """
        Emit the PostScript snippet 'ps' with all the attributes from 'gc'
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
        mightstroke = (gc.get_linewidth() > 0
                       and not _is_transparent(gc.get_rgb()))
        if not mightstroke:
            stroke = False
        if _is_transparent(rgbFace):
            fill = False
        hatch = gc.get_hatch()

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
            x, y, w, h = cliprect.bounds
            write('%1.4g %1.4g %1.4g %1.4g clipbox\n' % (w, h, x, y))
        clippath, clippath_trans = gc.get_clip_path()
        if clippath:
            id = self._get_clip_path(clippath, clippath_trans)
            write('%s\n' % id)

        # Jochen, is the strip necessary? - this could be a honking big string
        write(ps.strip())
        write("\n")

        if fill:
            if stroke or hatch:
                write("gsave\n")
            self.set_color(store=0, *rgbFace[:3])
            write("fill\n")
            if stroke or hatch:
                write("grestore\n")

        if hatch:
            hatch_name = self.create_hatch(hatch)
            write("gsave\n")
            write("%f %f %f " % gc.get_hatch_color()[:3])
            write("%s setpattern fill grestore\n" % hatch_name)

        if stroke:
            write("stroke\n")

        write("grestore\n")


def _is_transparent(rgb_or_rgba):
    if rgb_or_rgba is None:
        return True  # Consistent with rgbFace semantics.
    elif len(rgb_or_rgba) == 4:
        if rgb_or_rgba[3] == 0:
            return True
        if rgb_or_rgba[3] != 1:
            _log.warning(
                "The PostScript backend does not support transparency; "
                "partially transparent artists will be rendered opaque.")
        return False
    else:  # len() == 3.
        return False


class GraphicsContextPS(GraphicsContextBase):
    def get_capstyle(self):
        return {'butt': 0, 'round': 1, 'projecting': 2}[
            GraphicsContextBase.get_capstyle(self)]

    def get_joinstyle(self):
        return {'miter': 0, 'round': 1, 'bevel': 2}[
            GraphicsContextBase.get_joinstyle(self)]

    @cbook.deprecated("3.1")
    def shouldstroke(self):
        return (self.get_linewidth() > 0.0 and
                (len(self.get_rgb()) <= 3 or self.get_rgb()[3] != 0.0))


class _Orientation(Enum):
    portrait, landscape = range(2)

    def swap_if_landscape(self, shape):
        return shape[::-1] if self.name == "landscape" else shape


class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi = 72

    def draw(self):
        pass

    filetypes = {'ps': 'Postscript',
                 'eps': 'Encapsulated Postscript'}

    def get_default_filetype(self):
        return 'ps'

    def print_ps(self, outfile, *args, **kwargs):
        return self._print_ps(outfile, 'ps', *args, **kwargs)

    def print_eps(self, outfile, *args, **kwargs):
        return self._print_ps(outfile, 'eps', *args, **kwargs)

    def _print_ps(self, outfile, format, *args,
                  papertype=None, dpi=72, facecolor='w', edgecolor='w',
                  orientation='portrait',
                  **kwargs):
        if papertype is None:
            papertype = rcParams['ps.papersize']
        papertype = papertype.lower()
        cbook._check_in_list(['auto', *papersize], papertype=papertype)

        orientation = cbook._check_getitem(
            _Orientation, orientation=orientation.lower())

        self.figure.set_dpi(72)  # Override the dpi kwarg

        printer = (self._print_figure_tex
                   if rcParams['text.usetex'] else
                   self._print_figure)
        printer(outfile, format, dpi, facecolor, edgecolor,
                orientation, papertype, **kwargs)

    @cbook._delete_parameter("3.2", "dryrun")
    def _print_figure(
            self, outfile, format, dpi, facecolor, edgecolor,
            orientation, papertype, *,
            metadata=None, dryrun=False, bbox_inches_restore=None, **kwargs):
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

        metadata must be a dictionary. Currently, only the value for
        the key 'Creator' is used.
        """
        is_eps = format == 'eps'
        if isinstance(outfile, (str, os.PathLike)):
            outfile = title = os.fspath(outfile)
            title = title.encode("ascii", "replace").decode("ascii")
            passed_in_file_object = False
        elif is_writable_file_like(outfile):
            title = None
            passed_in_file_object = True
        else:
            raise ValueError("outfile must be a path or a file-like object")

        # find the appropriate papertype
        width, height = self.figure.get_size_inches()
        if papertype == 'auto':
            papertype = _get_papertype(
                *orientation.swap_if_landscape((width, height)))
        paper_width, paper_height = orientation.swap_if_landscape(
            papersize[papertype])

        if rcParams['ps.usedistiller']:
            # distillers improperly clip eps files if pagesize is too small
            if width > paper_width or height > paper_height:
                papertype = _get_papertype(
                    *orientation.swap_if_landscape(width, height))
                paper_width, paper_height = orientation.swap_if_landscape(
                    papersize[papertype])

        # center the figure on the paper
        xo = 72 * 0.5 * (paper_width - width)
        yo = 72 * 0.5 * (paper_height - height)

        l, b, w, h = self.figure.bbox.bounds
        llx = xo
        lly = yo
        urx = llx + w
        ury = lly + h
        rotation = 0
        if orientation is _Orientation.landscape:
            llx, lly, urx, ury = lly, llx, ury, urx
            xo, yo = 72 * paper_height - yo, xo
            rotation = 90
        bbox = (llx, lly, urx, ury)

        # generate PostScript code for the figure and store it in a string
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        if dryrun:
            class NullWriter:
                def write(self, *args, **kwargs):
                    pass

            self._pswriter = NullWriter()
        else:
            self._pswriter = StringIO()

        # mixed mode rendering
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(
            self.figure, width, height, dpi, ps_renderer,
            bbox_inches_restore=bbox_inches_restore)

        self.figure.draw(renderer)

        if dryrun:  # return immediately if dryrun (tightbbox=True)
            return

        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)

        # check for custom metadata
        if metadata is not None and 'Creator' in metadata:
            creator_str = metadata['Creator']
        else:
            creator_str = "matplotlib version " + __version__ + \
                ", http://matplotlib.org/"

        def print_figure_impl(fh):
            # write the PostScript headers
            if is_eps:
                print("%!PS-Adobe-3.0 EPSF-3.0", file=fh)
            else:
                print(f"%!PS-Adobe-3.0\n"
                      f"%%DocumentPaperSizes: {papertype}\n"
                      f"%%Pages: 1\n",
                      end="", file=fh)
            if title:
                print("%%Title: " + title, file=fh)
            # get source date from SOURCE_DATE_EPOCH, if set
            # See https://reproducible-builds.org/specs/source-date-epoch/
            source_date_epoch = os.getenv("SOURCE_DATE_EPOCH")
            if source_date_epoch:
                source_date = datetime.datetime.utcfromtimestamp(
                    int(source_date_epoch)).strftime("%a %b %d %H:%M:%S %Y")
            else:
                source_date = time.ctime()
            print(f"%%Creator: {creator_str}\n"
                  f"%%CreationDate: {source_date}\n"
                  f"%%Orientation: {orientation.name}\n"
                  f"%%BoundingBox: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                  f"%%EndComments\n",
                  end="", file=fh)

            Ndict = len(psDefs)
            print("%%BeginProlog", file=fh)
            if not rcParams['ps.useafm']:
                Ndict += len(ps_renderer.used_characters)
            print("/mpldict %d dict def" % Ndict, file=fh)
            print("mpldict begin", file=fh)
            for d in psDefs:
                d = d.strip()
                for l in d.split('\n'):
                    print(l.strip(), file=fh)
            if not rcParams['ps.useafm']:
                for font_filename, chars in \
                        ps_renderer.used_characters.values():
                    if len(chars):
                        font = get_font(font_filename)
                        glyph_ids = [font.get_char_index(c) for c in chars]

                        fonttype = rcParams['ps.fonttype']

                        # Can not use more than 255 characters from a
                        # single font for Type 3
                        if len(glyph_ids) > 255:
                            fonttype = 42

                        # The ttf to ps (subsetting) support doesn't work for
                        # OpenType fonts that are Postscript inside (like the
                        # STIX fonts).  This will simply turn that off to avoid
                        # errors.
                        if is_opentype_cff_font(font_filename):
                            raise RuntimeError(
                                "OpenType CFF fonts can not be saved using "
                                "the internal Postscript backend at this "
                                "time; consider using the Cairo backend")
                        else:
                            fh.flush()
                            try:
                                convert_ttf_to_ps(os.fsencode(font_filename),
                                                  fh, fonttype, glyph_ids)
                            except RuntimeError:
                                _log.warning("The PostScript backend does not "
                                             "currently support the selected "
                                             "font.")
                                raise
            print("end", file=fh)
            print("%%EndProlog", file=fh)

            if not is_eps:
                print("%%Page: 1 1", file=fh)
            print("mpldict begin", file=fh)

            print("%s translate" % _nums_to_str(xo, yo), file=fh)
            if rotation:
                print("%d rotate" % rotation, file=fh)
            print("%s clipbox" % _nums_to_str(width*72, height*72, 0, 0),
                  file=fh)

            # write the figure
            content = self._pswriter.getvalue()
            if not isinstance(content, str):
                content = content.decode('ascii')
            print(content, file=fh)

            # write the trailer
            print("end", file=fh)
            print("showpage", file=fh)
            if not is_eps:
                print("%%EOF", file=fh)
            fh.flush()

        if rcParams['ps.usedistiller']:
            # We are going to use an external program to process the output.
            # Write to a temporary file.
            with TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, "tmp.ps")
                with open(tmpfile, 'w', encoding='latin-1') as fh:
                    print_figure_impl(fh)
                if rcParams['ps.usedistiller'] == 'ghostscript':
                    gs_distill(tmpfile, is_eps, ptype=papertype, bbox=bbox)
                elif rcParams['ps.usedistiller'] == 'xpdf':
                    xpdf_distill(tmpfile, is_eps, ptype=papertype, bbox=bbox)
                _move_path_to_path_or_stream(tmpfile, outfile)

        else:
            # Write directly to outfile.
            if passed_in_file_object:
                requires_unicode = file_requires_unicode(outfile)

                if not requires_unicode:
                    fh = TextIOWrapper(outfile, encoding="latin-1")
                    # Prevent the TextIOWrapper from closing the underlying
                    # file.
                    fh.close = lambda: None
                else:
                    fh = outfile

                print_figure_impl(fh)
            else:
                with open(outfile, 'w', encoding='latin-1') as fh:
                    print_figure_impl(fh)

    @cbook._delete_parameter("3.2", "dryrun")
    def _print_figure_tex(
            self, outfile, format, dpi, facecolor, edgecolor,
            orientation, papertype, *,
            metadata=None, dryrun=False, bbox_inches_restore=None, **kwargs):
        """
        If text.usetex is True in rc, a temporary pair of tex/eps files
        are created to allow tex to manage the text layout via the PSFrags
        package. These files are processed to yield the final ps or eps file.

        metadata must be a dictionary. Currently, only the value for
        the key 'Creator' is used.
        """
        is_eps = format == 'eps'
        if is_writable_file_like(outfile):
            title = None
        else:
            try:
                title = os.fspath(outfile)
            except TypeError:
                raise ValueError(
                    "outfile must be a path or a file-like object")

        self.figure.dpi = 72  # ignore the dpi kwarg
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

        if dryrun:
            class NullWriter:
                def write(self, *args, **kwargs):
                    pass

            self._pswriter = NullWriter()
        else:
            self._pswriter = StringIO()

        # mixed mode rendering
        ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)
        renderer = MixedModeRenderer(self.figure,
                                     width, height, dpi, ps_renderer,
                                     bbox_inches_restore=bbox_inches_restore)

        self.figure.draw(renderer)

        if dryrun:  # return immediately if dryrun (tightbbox=True)
            return

        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)

        # check for custom metadata
        if metadata is not None and 'Creator' in metadata:
            creator_str = metadata['Creator']
        else:
            creator_str = "matplotlib version " + __version__ + \
                ", http://matplotlib.org/"

        # write to a temp file, we'll move it to outfile when done

        with TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tmp.ps")
            # get source date from SOURCE_DATE_EPOCH, if set
            # See https://reproducible-builds.org/specs/source-date-epoch/
            source_date_epoch = os.getenv("SOURCE_DATE_EPOCH")
            if source_date_epoch:
                source_date = datetime.datetime.utcfromtimestamp(
                    int(source_date_epoch)).strftime("%a %b %d %H:%M:%S %Y")
            else:
                source_date = time.ctime()
            pathlib.Path(tmpfile).write_text(
                f"""\
%!PS-Adobe-3.0 EPSF-3.0
{f'''%%Title: {title}
''' if title else ""}\
%%Creator: {creator_str}
%%CreationDate: {source_date}
%%BoundingBox: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}
%%EndComments
%%BeginProlog
/mpldict {len(psDefs)} dict def
mpldict begin
{"".join(psDefs)}
end
%%EndProlog
mpldict begin
{_nums_to_str(xo, yo)} translate
{_nums_to_str(width*72, height*72)} 0 0 clipbox
{self._pswriter.getvalue()}
end
showpage
""",
                encoding="latin-1")

            if orientation is _Orientation.landscape:  # now, ready to rotate
                width, height = height, width
                bbox = (lly, llx, ury, urx)

            # set the paper size to the figure size if is_eps. The
            # resulting ps file has the given size with correct bounding
            # box so that there is no need to call 'pstoeps'
            if is_eps:
                paper_width, paper_height = orientation.swap_if_landscape(
                    self.figure.get_size_inches())
            else:
                temp_papertype = _get_papertype(width, height)
                if papertype == 'auto':
                    papertype = temp_papertype
                    paper_width, paper_height = papersize[temp_papertype]
                else:
                    paper_width, paper_height = papersize[papertype]

            texmanager = ps_renderer.get_texmanager()
            font_preamble = texmanager.get_font_preamble()
            custom_preamble = texmanager.get_custom_preamble()

            psfrag_rotated = convert_psfrags(tmpfile, ps_renderer.psfrag,
                                             font_preamble,
                                             custom_preamble, paper_width,
                                             paper_height,
                                             orientation.name)

            if (rcParams['ps.usedistiller'] == 'ghostscript'
                    or rcParams['text.usetex']):
                gs_distill(tmpfile, is_eps, ptype=papertype, bbox=bbox,
                           rotated=psfrag_rotated)
            elif rcParams['ps.usedistiller'] == 'xpdf':
                xpdf_distill(tmpfile, is_eps, ptype=papertype, bbox=bbox,
                             rotated=psfrag_rotated)

            _move_path_to_path_or_stream(tmpfile, outfile)


def convert_psfrags(tmpfile, psfrags, font_preamble, custom_preamble,
                    paper_width, paper_height, orientation):
    """
    When we want to use the LaTeX backend with postscript, we write PSFrag tags
    to a temporary postscript file, each one marking a position for LaTeX to
    render some text. convert_psfrags generates a LaTeX document containing the
    commands to convert those tags to text. LaTeX/dvips produces the postscript
    file that includes the actual text.
    """
    with mpl.rc_context({
            "text.latex.preamble":
            mpl.rcParams["text.latex.preamble"] +
            r"\usepackage{psfrag,color}""\n"
            r"\usepackage[dvips]{graphicx}""\n"
            r"\geometry{papersize={%(width)sin,%(height)sin},"
            r"body={%(width)sin,%(height)sin},margin=0in}"
            % {"width": paper_width, "height": paper_height}
    }):
        dvifile = TexManager().make_dvi(
            "\n"
            r"\begin{figure}""\n"
            r"  \centering\leavevmode""\n"
            r"  %(psfrags)s""\n"
            r"  \includegraphics*[angle=%(angle)s]{%(epsfile)s}""\n"
            r"\end{figure}"
            % {
                "psfrags": "\n".join(psfrags),
                "angle": 90 if orientation == 'landscape' else 0,
                "epsfile": pathlib.Path(tmpfile).resolve().as_posix(),
            },
            fontsize=10)  # tex's default fontsize.

    with TemporaryDirectory() as tmpdir:
        psfile = os.path.join(tmpdir, "tmp.ps")
        cbook._check_and_log_subprocess(
            ['dvips', '-q', '-R0', '-o', psfile, dvifile], _log)
        shutil.move(psfile, tmpfile)

    # check if the dvips created a ps in landscape paper.  Somehow,
    # above latex+dvips results in a ps file in a landscape mode for a
    # certain figure sizes (e.g., 8.3in, 5.8in which is a5). And the
    # bounding box of the final output got messed up. We check see if
    # the generated ps file is in landscape and return this
    # information. The return value is used in pstoeps step to recover
    # the correct bounding box. 2010-06-05 JJL
    with open(tmpfile) as fh:
        psfrag_rotated = "Landscape" in fh.read(1000)
    return psfrag_rotated


def gs_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's pswrite or epswrite device to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. The output is low-level, converting text to outlines.
    """

    if eps:
        paper_option = "-dEPSCrop"
    else:
        paper_option = "-sPAPERSIZE=%s" % ptype

    psfile = tmpfile + '.ps'
    dpi = rcParams['ps.distiller.res']

    cbook._check_and_log_subprocess(
        [mpl._get_executable_info("gs").executable,
         "-dBATCH", "-dNOPAUSE", "-r%d" % dpi, "-sDEVICE=ps2write",
         paper_option, "-sOutputFile=%s" % psfile, tmpfile],
        _log)

    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)

    # While it is best if above steps preserve the original bounding
    # box, there seem to be cases when it is not. For those cases,
    # the original bbox can be restored during the pstoeps step.

    if eps:
        # For some versions of gs, above steps result in an ps file where the
        # original bbox is no more correct. Do not adjust bbox for now.
        pstoeps(tmpfile, bbox, rotated=rotated)


def xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. This distiller is preferred, generating high-level postscript
    output that treats text as text.
    """
    pdffile = tmpfile + '.pdf'
    psfile = tmpfile + '.ps'

    # Pass options as `-foo#bar` instead of `-foo=bar` to keep Windows happy
    # (https://www.ghostscript.com/doc/9.22/Use.htm#MS_Windows).
    cbook._check_and_log_subprocess(
        ["ps2pdf",
         "-dAutoFilterColorImages#false",
         "-dAutoFilterGrayImages#false",
         "-sAutoRotatePages#None",
         "-sGrayImageFilter#FlateEncode",
         "-sColorImageFilter#FlateEncode",
         "-dEPSCrop" if eps else "-sPAPERSIZE#%s" % ptype,
         tmpfile, pdffile], _log)
    cbook._check_and_log_subprocess(
        ["pdftops", "-paper", "match", "-level2", pdffile, psfile], _log)

    os.remove(tmpfile)
    shutil.move(psfile, tmpfile)

    if eps:
        pstoeps(tmpfile)

    for fname in glob.glob(tmpfile+'.*'):
        os.remove(fname)


def get_bbox_header(lbrt, rotated=False):
    """
    return a postscript header string for the given bbox lbrt=(l, b, r, t).
    Optionally, return rotate command.
    """

    l, b, r, t = lbrt
    if rotated:
        rotate = "%.2f %.2f translate\n90 rotate" % (l+r, 0)
    else:
        rotate = ""
    bbox_info = '%%%%BoundingBox: %d %d %d %d' % (l, b, np.ceil(r), np.ceil(t))
    hires_bbox_info = '%%%%HiResBoundingBox: %.6f %.6f %.6f %.6f' % (
        l, b, r, t)

    return '\n'.join([bbox_info, hires_bbox_info]), rotate


def pstoeps(tmpfile, bbox=None, rotated=False):
    """
    Convert the postscript to encapsulated postscript.  The bbox of
    the eps file will be replaced with the given *bbox* argument. If
    None, original bbox will be used.
    """

    # if rotated==True, the output eps file need to be rotated
    if bbox:
        bbox_info, rotate = get_bbox_header(bbox, rotated=rotated)
    else:
        bbox_info, rotate = None, None

    epsfile = tmpfile + '.eps'
    with open(epsfile, 'wb') as epsh, open(tmpfile, 'rb') as tmph:
        write = epsh.write
        # Modify the header:
        for line in tmph:
            if line.startswith(b'%!PS'):
                write(b"%!PS-Adobe-3.0 EPSF-3.0\n")
                if bbox:
                    write(bbox_info.encode('ascii') + b'\n')
            elif line.startswith(b'%%EndComments'):
                write(line)
                write(b'%%BeginProlog\n'
                      b'save\n'
                      b'countdictstack\n'
                      b'mark\n'
                      b'newpath\n'
                      b'/showpage {} def\n'
                      b'/setpagedevice {pop} def\n'
                      b'%%EndProlog\n'
                      b'%%Page 1 1\n')
                if rotate:
                    write(rotate.encode('ascii') + b'\n')
                break
            elif bbox and line.startswith((b'%%Bound', b'%%HiResBound',
                                           b'%%DocumentMedia', b'%%Pages')):
                pass
            else:
                write(line)
        # Now rewrite the rest of the file, and modify the trailer.
        # This is done in a second loop such that the header of the embedded
        # eps file is not modified.
        for line in tmph:
            if line.startswith(b'%%EOF'):
                write(b'cleartomark\n'
                      b'countdictstack\n'
                      b'exch sub { end } repeat\n'
                      b'restore\n'
                      b'showpage\n'
                      b'%%EOF\n')
            elif line.startswith(b'%%PageBoundingBox'):
                pass
            else:
                write(line)

    os.remove(tmpfile)
    shutil.move(epsfile, tmpfile)


FigureManagerPS = FigureManagerBase


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


@_Backend.export
class _BackendPS(_Backend):
    FigureCanvas = FigureCanvasPS
