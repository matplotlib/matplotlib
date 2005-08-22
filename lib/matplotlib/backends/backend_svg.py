from __future__ import division

import os, codecs, base64, tempfile

from matplotlib import verbose, __version__
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import math_parse_s_ft2font_svg

backend_version = __version__

def new_figure_manager(num, *args):
    thisFig = Figure(*args)
    canvas  = FigureCanvasSVG(thisFig)
    manager = FigureManagerSVG(canvas, num)
    return manager


_fontd = {}
_capstyle_d = {'projecting' : 'square', 'butt' : 'butt', 'round': 'round',}
class RendererSVG(RendererBase):
    def __init__(self, width, height, svgwriter):
        self.width=width
        self.height=height
        self._svgwriter = svgwriter

        self._groupd = {}
        self._clipd = {}
        svgwriter.write(svgProlog%(width,height,width,height))

    def _draw_svg_element(self, element, details, gc, rgbFace):
        cliprect, clipid = self._get_gc_clip_svg(gc)
        if clipid is None:
            clippath = ''
        else:
            clippath = 'clip-path:url(#%s);' % clipid

        self._svgwriter.write ('%s<%s %s %s/>\n' % (
            cliprect,
            element, self._get_style(gc, rgbFace, clippath), details))

    def _get_font(self, prop):
        key = hash(prop)
        font = _fontd.get(key)
        if font is None:
            fname = fontManager.findfont(prop)
            font = FT2Font(str(fname))
            _fontd[key] = font
        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, 72.0)
        return font

    def _get_style(self, gc, rgbFace, clippath):
        """
        return the style string.
        style is generated from the GraphicsContext, rgbFace and clippath
        """
        if rgbFace is None:
            fill = 'none'
        else:
            fill = rgb2hex(rgbFace)

        offset, seq = gc.get_dashes()
        if seq is None:
            dashes = ''
        else:
            dashes = 'stroke-dasharray: %s; stroke-dashoffset: %f;' % (
                ' '.join(['%f'%val for val in seq]), offset)

        return 'style="fill: %s; stroke: %s; stroke-width: %f; ' \
               'stroke-linejoin: %s; stroke-linecap: %s; %s opacity: %f; ' \
               '%s"' % (
                   fill,
                   rgb2hex(gc.get_rgb()),
                   gc.get_linewidth(),
                   gc.get_joinstyle(),
                   _capstyle_d[gc.get_capstyle()],
                   dashes,
                   gc.get_alpha(),
                   clippath,
                   )

    def _get_gc_clip_svg(self, gc):
        cliprect = gc.get_clip_rectangle()
        if cliprect is None:
            return '', None
        else:
            # See if we've already seen this clip rectangle
            key = hash(cliprect)
            if self._clipd.get(key) is None:  # If not, store a new clipPath
                self._clipd[key] = cliprect
                x, y, w, h = cliprect
                y = self.height-(y+h)
                box = """\
<defs>
    <clipPath id="%(key)s">
    <rect x="%(x)f" y="%(y)f" width="%(w)f" height="%(h)f"
    style="stroke: gray; fill: none;"/>
    </clipPath>
</defs>
""" % locals()
                return box, key
            else:
                # return id of previously defined clipPath
                return '', key

    def open_group(self, s):
        self._groupd[s] = self._groupd.get(s,0) + 1
        self._svgwriter.write('<g id="%s%d">\n' % (s, self._groupd[s]))

    def close_group(self, s):
        self._svgwriter.write('</g>\n')

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2):
        """
        Currently implemented by drawing a circle of diameter width, not an
        arc. angle1, angle2 not used
        """
        details = 'cx="%f" cy="%f" r="%f"' % (x,self.height-y,width/2)
        self._draw_svg_element('circle', details, gc, rgbFace)

    def draw_image(self, x, y, im, bbox):
        filename = os.path.join (tempfile.gettempdir(),
                                 tempfile.gettempprefix() + '.png'
                                 )

        verbose.report ('Writing image file for include: %s' % filename)
        # im.write_png() accepts a filename, not file object, would be
        # good to avoid using files and write to mem with StringIO

        # JDH: it *would* be good, but I don't know how to do this
        # since libpng seems to want a FILE* and StringIO doesn't seem
        # to provide one.  I suspect there is a way, but I don't know
        # it

        im.flipud_out()

        h,w = im.get_size_out()
        y = self.height-y-h 
        im.write_png(filename) 

	imfile = file (filename, 'r')
	image64 = base64.b64encode (imfile.read())
	imfile.close()
	os.remove(filename)
        lines = [image64[i:i+76] for i in range(0, len(image64), 76)]

        self._svgwriter.write (
            '<image x="%f" y="%f" width="%f" height="%f" '
            'xlink:href="data:image/png;base64,\n%s" />\n'
            % (x, y, w+1, h+1, '\n'.join(lines))
            )

         # unflip
        im.flipud_out()

    def draw_line(self, gc, x1, y1, x2, y2):
        details = 'd="M %f,%f L %f,%f"' % (x1, self.height-y1,
                                           x2, self.height-y2)
        self._draw_svg_element('path', details, gc, None)

    def draw_lines(self, gc, x, y, transform=None):
        if len(x)==0: return
        if len(x)!=len(y):
            raise ValueError('x and y must be the same length')

        y = self.height - y
        details = ['d="M %f,%f' % (x[0], y[0])]
        xys = zip(x[1:], y[1:])
        details.extend(['L %f,%f' % tup for tup in xys])
        details.append('"')
        details = ' '.join(details)
        self._draw_svg_element('path', details, gc, None)

    def draw_point(self, gc, x, y):
        # result seems to have a hole in it...
        self.draw_arc(gc, gc.get_rgb(), x, y, 1, 0, 0, 0)

    def draw_polygon(self, gc, rgbFace, points):
        details = 'points = "%s"' % ' '.join(['%f,%f'%(x,self.height-y)
                                              for x, y in points])
        self._draw_svg_element('polygon', details, gc, rgbFace)

    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        details = 'width="%f" height="%f" x="%f" y="%f"' % (width, height, x,
                                                         self.height-y-height)
        self._draw_svg_element('rect', details, gc, rgbFace)

    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        if ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)
            return

        font = self._get_font(prop)

        thetext = '%s' % s
        fontfamily=font.family_name
        fontstyle=font.style_name
        fontsize = prop.get_size_in_points()
        color = rgb2hex(gc.get_rgb())

        style = 'font-size: %f; font-family: %s; font-style: %s; fill: %s;'%(fontsize, fontfamily,fontstyle, color)
        if angle!=0:
            transform = 'transform="translate(%f,%f) rotate(%1.1f) translate(%f,%f)"' % (x,y,-angle,-x,-y) # Inkscape doesn't support rotate(angle x y)
        else: transform = ''

        svg = """\
<text style="%(style)s" x="%(x)f" y="%(y)f" %(transform)s>%(thetext)s</text>
""" % locals()
        self._svgwriter.write (svg)

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        """
        Draw math text using matplotlib.mathtext
        """
        fontsize = prop.get_size_in_points()
        width, height, svg_glyphs = math_parse_s_ft2font_svg(s, 72, fontsize)
        color = rgb2hex(gc.get_rgb())

        svg = ""
        self.open_group("mathtext")
        for fontname, fontsize, num, ox, oy, metrics in svg_glyphs:
            thetext=unichr(num)
            thetext.encode('utf-8')
            style = 'font-size: %f; font-family: %s; fill: %s;'%(fontsize, fontname, color)
            if angle!=0:
                transform = 'transform="translate(%f,%f) rotate(%1.1f) translate(%f,%f)"' % (x,y,-angle,-x,-y) # Inkscape doesn't support rotate(angle x y)
            else: transform = ''
            newx, newy = x+ox, y-oy
            svg += """\
<text style="%(style)s" x="%(newx)f" y="%(newy)f" %(transform)s>%(thetext)s</text>
""" % locals()

        self._svgwriter.write (svg)
        self.close_group("mathtext")

    def finish(self):
        self._svgwriter.write('</svg>\n')

    def flipy(self):
        return True

    def get_canvas_width_height(self):
        return self.width, self.height

    def get_text_width_height(self, s, prop, ismath):
        if ismath:
            width, height, glyphs = math_parse_s_ft2font_svg(
                s, 72, prop.get_size_in_points())
            return width, height
        font = self._get_font(prop)
        font.set_text(s, 0.0)
        w, h = font.get_width_height()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        return w, h


class FigureCanvasSVG(FigureCanvasBase):

    def print_figure(self, filename, dpi=80,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):
        # save figure settings
        origDPI       = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        self.figure.dpi.set(72)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)
        width, height = self.figure.get_size_inches()
        w, h = width*72, height*72

        basename, ext = os.path.splitext(filename)
        if not len(ext): filename += '.svg'
        svgwriter = codecs.open( filename, 'w', 'utf-8' )
        renderer = RendererSVG(w, h, svgwriter)
        self.figure.draw(renderer)
        renderer.finish()

        # restore figure settings
        self.figure.dpi.set(origDPI)
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        svgwriter.close()

class FigureManagerSVG(FigureManagerBase):
    pass

FigureManager = FigureManagerSVG

svgProlog = """<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN"
"http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
<!-- Created with matplotlib (http://matplotlib.sourceforge.net/) -->
<svg
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   version="1.1"
   width="%i" height="%i" viewBox="0 0 %i %i"
   id="svg1">
"""
