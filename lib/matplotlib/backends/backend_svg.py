from __future__ import division

import os, codecs, base64, tempfile, urllib, gzip

from matplotlib import agg
from matplotlib import verbose, __version__, rcParams
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import is_string_like, is_writable_file_like, maxdict
from matplotlib.colors import rgb2hex
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT, LOAD_NO_HINTING
from matplotlib.mathtext import MathTextParser

from xml.sax.saxutils import escape as escape_xml_text

backend_version = __version__

def new_figure_manager(num, *args, **kwargs):
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args)
    canvas  = FigureCanvasSVG(thisFig)
    manager = FigureManagerSVG(canvas, num)
    return manager


_capstyle_d = {'projecting' : 'square', 'butt' : 'butt', 'round': 'round',}
class RendererSVG(RendererBase):
    FONT_SCALE = 100.0
    fontd = maxdict(50)

    def __init__(self, width, height, svgwriter, basename=None):
        self.width=width
        self.height=height
        self._svgwriter = svgwriter

        self._groupd = {}
        if not rcParams['svg.image_inline']:
            assert basename is not None
            self.basename = basename
            self._imaged = {}
        self._clipd = {}
        self._char_defs = {}
        self.mathtext_parser = MathTextParser('SVG')
        svgwriter.write(svgProlog%(width,height,width,height))

    def _draw_svg_element(self, element, details, gc, rgbFace):
        cliprect, clipid = self._get_gc_clip_svg(gc)
        if clipid is None:
            clippath = ''
        else:
            clippath = 'clip-path="url(#%s)"' % clipid

        style = self._get_style(gc, rgbFace)
        self._svgwriter.write ('%s<%s style="%s" %s %s/>\n' % (
            cliprect, element, style, clippath, details))

    def _get_font(self, prop):
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

    def _get_style(self, gc, rgbFace):
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
            dashes = 'stroke-dasharray: %s; stroke-dashoffset: %s;' % (
                ','.join(['%s'%val for val in seq]), offset)

        linewidth = gc.get_linewidth()
        if linewidth:
            return 'fill: %s; stroke: %s; stroke-width: %s; ' \
                'stroke-linejoin: %s; stroke-linecap: %s; %s opacity: %s' % (
                         fill,
                         rgb2hex(gc.get_rgb()),
                         linewidth,
                         gc.get_joinstyle(),
                         _capstyle_d[gc.get_capstyle()],
                         dashes,
                         gc.get_alpha(),
                )
        else:
            return 'fill: %s; opacity: %s' % (\
                         fill,
                         gc.get_alpha(),
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
                style = "stroke: gray; fill: none;"
                box = """\
<defs>
    <clipPath id="%(key)s">
    <rect x="%(x)s" y="%(y)s" width="%(w)s" height="%(h)s"
    style="%(style)s"/>
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

    def draw_path(self, gc, rgbFace, path):
        cmd = []

        while 1:
            code, xp, yp = path.vertex()
            yp = self.height - yp

            if code == agg.path_cmd_stop:
                cmd.append('z') # Hack, path_cmd_end_poly not found
                break
            elif code == agg.path_cmd_move_to:
                cmd.append('M%g %g' % (xp, yp))
            elif code == agg.path_cmd_line_to:
                cmd.append('L%g %g' % (xp, yp))
            elif code == agg.path_cmd_curve3:
                verts = [xp, yp]
                verts.extent(path.vertex()[1:])
                verts[-1] = self.height - verts[-1]
                cmd.append('Q%g %g %g %g' % tuple(verts))
            elif code == agg.path_cmd_curve4:
                verts = [xp, yp]
                verts.extend(path.vertex()[1:])
                verts[-1] = self.height - verts[-1]
                verts.extend(path.vertex()[1:])
                verts[-1] = self.height - verts[-1]
                cmd.append('C%g %g %g %g %g %g'%tuple(verts))
            elif code == agg.path_cmd_end_poly:
                cmd.append('z')

        path_data = "".join(cmd)
        self._draw_svg_element("path", 'd="%s"' % path_data, gc, rgbFace)

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2, rotation):
        """
        Ignores angles for now
        """
        details = 'cx="%s" cy="%s" rx="%s" ry="%s" transform="rotate(%1.1f %s %s)"' % \
            (x,  self.height-y, width/2.0, height/2.0, -rotation, x, self.height-y)
        self._draw_svg_element('ellipse', details, gc, rgbFace)

    def option_image_nocomposite(self):
        """
        if svg.image_noscale is True, compositing multiple images into one is prohibited
        """
        return rcParams['svg.image_noscale']

    def draw_image(self, x, y, im, bbox):
        trans = [1,0,0,1,0,0]
        transstr = ''
        if rcParams['svg.image_noscale']:
            trans = list(im.get_matrix())
            if im.get_interpolation() != 0:
                trans[4] += trans[0]
                trans[5] += trans[3]
            trans[5] = -trans[5]
            transstr = 'transform="matrix(%s %s %s %s %s %s)" '%tuple(trans)
            assert trans[1] == 0
            assert trans[2] == 0
            numrows,numcols = im.get_size()
            im.reset_matrix()
            im.set_interpolation(0)
            im.resize(numcols, numrows)

        h,w = im.get_size_out()

        if rcParams['svg.image_inline']:
            filename = os.path.join (tempfile.gettempdir(),
                                    tempfile.gettempprefix() + '.png'
                                    )

            verbose.report ('Writing temporary image file for inlining: %s' % filename)
            # im.write_png() accepts a filename, not file object, would be
            # good to avoid using files and write to mem with StringIO

            # JDH: it *would* be good, but I don't know how to do this
            # since libpng seems to want a FILE* and StringIO doesn't seem
            # to provide one.  I suspect there is a way, but I don't know
            # it

            im.flipud_out()
            im.write_png(filename)
            im.flipud_out()

            imfile = file (filename, 'rb')
            image64 = base64.encodestring (imfile.read())
            imfile.close()
            os.remove(filename)
            hrefstr = 'data:image/png;base64,\n' + image64

        else:
            self._imaged[self.basename] = self._imaged.get(self.basename,0) + 1
            filename = '%s.image%d.png'%(self.basename, self._imaged[self.basename])
            verbose.report( 'Writing image file for inclusion: %s' % filename)
            im.flipud_out()
            im.write_png(filename)
            im.flipud_out()
            hrefstr = filename

        self._svgwriter.write (
            '<image x="%s" y="%s" width="%s" height="%s" '
            'xlink:href="%s" %s/>\n'%(x/trans[0], (self.height-y)/trans[3]-h, w, h, hrefstr, transstr)
            )

    def draw_line(self, gc, x1, y1, x2, y2):
        details = 'd="M%s,%sL%s,%s"' % (x1, self.height-y1,
                                           x2, self.height-y2)
        self._draw_svg_element('path', details, gc, None)

    def draw_lines(self, gc, x, y, transform=None):
        if len(x)==0: return
        if len(x)!=len(y):
            raise ValueError('x and y must be the same length')

        y = self.height - y
        details = ['d="M%s,%s' % (x[0], y[0])]
        xys = zip(x[1:], y[1:])
        details.extend(['L%s,%s' % tup for tup in xys])
        details.append('"')
        details = ''.join(details)
        self._draw_svg_element('path', details, gc, None)

    def draw_point(self, gc, x, y):
        # result seems to have a hole in it...
        self.draw_arc(gc, gc.get_rgb(), x, y, 1, 0, 0, 0, 0)

    def draw_polygon(self, gc, rgbFace, points):
        details = 'points = "%s"' % ' '.join(['%s,%s'%(x,self.height-y)
                                              for x, y in points])
        self._draw_svg_element('polygon', details, gc, rgbFace)

    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        details = 'width="%s" height="%s" x="%s" y="%s"' % (width, height, x,
                                                            self.height-y-height)
        self._draw_svg_element('rect', details, gc, rgbFace)

    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        if ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)
            return

        font = self._get_font(prop)
        font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
        y -= font.get_descent() / 64.0

        fontsize = prop.get_size_in_points()
        color = rgb2hex(gc.get_rgb())

        if rcParams['svg.embed_char_paths']:
            svg = ['<g style="fill: %s; opacity: %s" transform="' % (color, gc.get_alpha())]
            if angle != 0:
                svg.append('translate(%s,%s)rotate(%1.1f)' % (x,y,-angle))
            elif x != 0 or y != 0:
                svg.append('translate(%s,%s)' % (x, y))
            svg.append('scale(%s)">\n' % (fontsize / self.FONT_SCALE))

            cmap = font.get_charmap()
            lastgind = None
            currx = 0
            for c in s:
                charid = self._add_char_def(prop, c)
                ccode = ord(c)
                gind = cmap.get(ccode)
                if gind is None:
                    ccode = ord('?')
                    gind = 0
                glyph = font.load_char(ccode, flags=LOAD_NO_HINTING)

                if lastgind is not None:
                    kern = font.get_kerning(lastgind, gind, KERNING_DEFAULT)
                else:
                    kern = 0
                lastgind = gind
                currx += kern/64.0

                svg.append('<use xlink:href="#%s"' % charid)
                if currx != 0:
                    svg.append(' transform="translate(%s)"' %
                               (currx * (self.FONT_SCALE / fontsize)))
                svg.append('/>\n')
                currx += (glyph.linearHoriAdvance / 65536.0) / (self.FONT_SCALE / fontsize)
            svg.append('</g>\n')
            svg = ''.join(svg)
        else:
            thetext = escape_xml_text(s)
            fontfamily = font.family_name
            fontstyle = prop.get_style()

            style = ('font-size: %f; font-family: %s; font-style: %s; fill: %s; opacity: %s' %
                     (fontsize, fontfamily,fontstyle, color, gc.get_alpha()))
            if angle!=0:
                transform = 'transform="translate(%s,%s) rotate(%1.1f) translate(%s,%s)"' % (x,y,-angle,-x,-y)
                # Inkscape doesn't support rotate(angle x y)
            else:
                transform = ''

            svg = """\
<text style="%(style)s" x="%(x)s" y="%(y)s" %(transform)s>%(thetext)s</text>
""" % locals()
        self._svgwriter.write (svg)

    def _add_char_def(self, prop, char):
        if isinstance(prop, FontProperties):
            newprop = prop.copy()
            font = self._get_font(newprop)
        else:
            font = prop
        font.set_size(self.FONT_SCALE, 72)
        ps_name = font.get_sfnt()[(1,0,0,6)]
        char_id = urllib.quote('%s-%d' % (ps_name, ord(char)))
        char_num, path = self._char_defs.get(char_id, (None, None))
        if char_num is not None:
            return char_num

        path_data = []
        glyph = font.load_char(ord(char), flags=LOAD_NO_HINTING)
        currx, curry = 0.0, 0.0
        for step in glyph.path:
            if step[0] == 0:   # MOVE_TO
                path_data.append("M%s %s" %
                                 (step[1], -step[2]))
            elif step[0] == 1: # LINE_TO
                path_data.append("l%s %s" %
                                 (step[1] - currx, -step[2] - curry))
            elif step[0] == 2: # CURVE3
                path_data.append("q%s %s %s %s" %
                                 (step[1] - currx, -step[2] - curry,
                                  step[3] - currx, -step[4] - curry))
            elif step[0] == 3: # CURVE4
                path_data.append("c%s %s %s %s %s %s" %
                                 (step[1] - currx, -step[2] - curry,
                                  step[3] - currx, -step[4] - curry,
                                  step[5] - currx, -step[6] - curry))
            elif step[0] == 4: # ENDPOLY
                path_data.append("z")
                currx, curry = 0.0, 0.0

            if step[0] != 4:
                currx, curry = step[-2], -step[-1]
        char_num = 'c_%x' % len(self._char_defs)
        path_element = '<path id="%s" d="%s"/>\n' % (char_num, ''.join(path_data))
        self._char_defs[char_id] = (char_num, path_element)
        return char_num

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        """
        Draw math text using matplotlib.mathtext
        """
        width, height, descent, svg_elements, used_characters = \
            self.mathtext_parser.parse(s, 72, prop)
        svg_glyphs = svg_elements.svg_glyphs
        svg_rects = svg_elements.svg_rects
        color = rgb2hex(gc.get_rgb())

        self.open_group("mathtext")

        style = "fill: %s" % color

        if rcParams['svg.embed_char_paths']:
            svg = ['<g style="%s" transform="' % style]
            if angle != 0:
                svg.append('translate(%s,%s)rotate(%1.1f)'
                           % (x,y,-angle) )
            else:
                svg.append('translate(%s,%s)' % (x, y))
            svg.append('">\n')

            for font, fontsize, thetext, new_x, new_y_mtc, metrics in svg_glyphs:
                charid = self._add_char_def(font, thetext)

                svg.append('<use xlink:href="#%s" transform="translate(%s,%s)scale(%s)"/>\n' %
                           (charid, new_x, -new_y_mtc, fontsize / self.FONT_SCALE))
            svg.append('</g>\n')
        else: # not rcParams['svg.embed_char_paths']
            svg = ['<text style="%s" x="%f" y="%f"' % (style, x, y)]

            if angle != 0:
                svg.append(' transform="translate(%f,%f) rotate(%1.1f) translate(%f,%f)"'
                           % (x,y,-angle,-x,-y) ) # Inkscape doesn't support rotate(angle x y)
            svg.append('>\n')

            curr_x,curr_y = 0.0,0.0

            for font, fontsize, thetext, new_x, new_y_mtc, metrics in svg_glyphs:
                new_y = - new_y_mtc
                style = "font-size: %f; font-family: %s" % (fontsize, font.family_name)

                svg.append('<tspan style="%s"' % style)
                xadvance = metrics.advance
                svg.append(' textLength="%s"' % xadvance)

                dx = new_x - curr_x
                if dx != 0.0:
                    svg.append(' dx="%s"' % dx)

                dy = new_y - curr_y
                if dy != 0.0:
                    svg.append(' dy="%s"' % dy)

                thetext = escape_xml_text(thetext)

                svg.append('>%s</tspan>\n' % thetext)

                curr_x = new_x + xadvance
                curr_y = new_y

            svg.append('</text>\n')

        if len(svg_rects):
            style = "fill: %s; stroke: none" % color
            svg.append('<g style="%s" transform="' % style)
            if angle != 0:
                svg.append('translate(%s,%s) rotate(%1.1f)'
                           % (x,y,-angle) )
            else:
                svg.append('translate(%s,%s)' % (x, y))
            svg.append('">\n')

            for x, y, width, height in svg_rects:
                svg.append('<rect x="%s" y="%s" width="%s" height="%s" fill="black" stroke="none" />' % (x, -y + height, width, height))
            svg.append("</g>")

        self._svgwriter.write (''.join(svg))
        self.close_group("mathtext")

    def finish(self):
        write = self._svgwriter.write
        if len(self._char_defs):
            write('<defs id="fontpaths">\n')
            for char_num, path in self._char_defs.values():
                write(path)
            write('</defs>\n')
        write('</svg>\n')

    def flipy(self):
        return True

    def get_canvas_width_height(self):
        return self.width, self.height

    def get_text_width_height_descent(self, s, prop, ismath):
        if ismath:
            width, height, descent, trash, used_characters = \
                self.mathtext_parser.parse(s, 72, prop)
            return width, height, descent
        font = self._get_font(prop)
        font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
        w, h = font.get_width_height()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        d = font.get_descent()
        d /= 64.0
        return w, h, d


class FigureCanvasSVG(FigureCanvasBase):
    filetypes = {'svg': 'Scalable Vector Graphics',
                 'svgz': 'Scalable Vector Graphics'}

    def print_svg(self, filename, *args, **kwargs):
        if is_string_like(filename):
            fh_to_close = svgwriter = codecs.open(filename, 'w', 'utf-8')
        elif is_writable_file_like(filename):
            svgwriter = codecs.EncodedFile(filename, 'utf-8')
            fh_to_close = None
        else:
            raise ValueError("filename must be a path or a file-like object")
        return self._print_svg(filename, svgwriter, fh_to_close)

    def print_svgz(self, filename, *args, **kwargs):
        if is_string_like(filename):
            gzipwriter = gzip.GzipFile(filename, 'w')
            fh_to_close = svgwriter = codecs.EncodedFile(gzipwriter, 'utf-8')
        elif is_writable_file_like(filename):
            fh_to_close = gzipwriter = gzip.GzipFile(fileobj=filename, mode='w')
            svgwriter = codecs.EncodedFile(gzipwriter, 'utf-8')
        else:
            raise ValueError("filename must be a path or a file-like object")
        return self._print_svg(filename, svgwriter, fh_to_close)

    def _print_svg(self, filename, svgwriter, fh_to_close=None):
        self.figure.dpi.set(72)
        width, height = self.figure.get_size_inches()
        w, h = width*72, height*72

        renderer = RendererSVG(w, h, svgwriter, filename)
        self.figure.draw(renderer)
        renderer.finish()
        if fh_to_close is not None:
            svgwriter.close()

    def get_default_filetype(self):
        return 'svg'

class FigureManagerSVG(FigureManagerBase):
    pass

FigureManager = FigureManagerSVG

svgProlog = """\
<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Created with matplotlib (http://matplotlib.sourceforge.net/) -->
<svg width="%ipt" height="%ipt" viewBox="0 0 %i %i"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   version="1.1"
   id="svg1">
"""
