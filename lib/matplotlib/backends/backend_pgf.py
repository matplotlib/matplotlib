import atexit
import codecs
import errno
import logging
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import weakref

import matplotlib as mpl
from matplotlib import _png, cbook, font_manager as fm, __version__, rcParams
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
    RendererBase)
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.cbook import is_writable_file_like
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf

_log = logging.getLogger(__name__)


###############################################################################


@cbook.deprecated("3.0")
def get_texcommand():
    """Get chosen TeX system from rc."""
    texsystem_options = ["xelatex", "lualatex", "pdflatex"]
    texsystem = rcParams["pgf.texsystem"]
    return texsystem if texsystem in texsystem_options else "xelatex"


def get_fontspec():
    """Build fontspec preamble from rc."""
    latex_fontspec = []
    texcommand = rcParams["pgf.texsystem"]

    if texcommand != "pdflatex":
        latex_fontspec.append("\\usepackage{fontspec}")

    if texcommand != "pdflatex" and rcParams["pgf.rcfonts"]:
        families = ["serif", "sans\\-serif", "monospace"]
        commands = ["setmainfont", "setsansfont", "setmonofont"]
        for family, command in zip(families, commands):
            # 1) Forward slashes also work on Windows, so don't mess with
            # backslashes.  2) The dirname needs to include a separator.
            path = pathlib.Path(fm.findfont(family))
            latex_fontspec.append(r"\%s{%s}[Path=%s]" % (
                command, path.name, path.parent.as_posix() + "/"))

    return "\n".join(latex_fontspec)


def get_preamble():
    """Get LaTeX preamble from rc."""
    return "\n".join(rcParams["pgf.preamble"])

###############################################################################

# This almost made me cry!!!
# In the end, it's better to use only one unit for all coordinates, since the
# arithmetic in latex seems to produce inaccurate conversions.
latex_pt_to_in = 1. / 72.27
latex_in_to_pt = 1. / latex_pt_to_in
mpl_pt_to_in = 1. / 72.
mpl_in_to_pt = 1. / mpl_pt_to_in

###############################################################################
# helper functions

NO_ESCAPE = r"(?<!\\)(?:\\\\)*"
re_mathsep = re.compile(NO_ESCAPE + r"\$")
re_escapetext = re.compile(NO_ESCAPE + "([_^$%])")
repl_escapetext = lambda m: "\\" + m.group(1)
re_mathdefault = re.compile(NO_ESCAPE + r"(\\mathdefault)")
repl_mathdefault = lambda m: m.group(0)[:-len(m.group(1))]


def common_texification(text):
    """
    Do some necessary and/or useful substitutions for texts to be included in
    LaTeX documents.
    """

    # Sometimes, matplotlib adds the unknown command \mathdefault.
    # Not using \mathnormal instead since this looks odd for the latex cm font.
    text = re_mathdefault.sub(repl_mathdefault, text)

    # split text into normaltext and inline math parts
    parts = re_mathsep.split(text)
    for i, s in enumerate(parts):
        if not i % 2:
            # textmode replacements
            s = re_escapetext.sub(repl_escapetext, s)
        else:
            # mathmode replacements
            s = r"\(\displaystyle %s\)" % s
        parts[i] = s

    return "".join(parts)


def writeln(fh, line):
    # every line of a file included with \\input must be terminated with %
    # if not, latex will create additional vertical spaces for some reason
    fh.write(line)
    fh.write("%\n")


def _font_properties_str(prop):
    # translate font properties to latex commands, return as string
    commands = []

    families = {"serif": r"\rmfamily", "sans": r"\sffamily",
                "sans-serif": r"\sffamily", "monospace": r"\ttfamily"}
    family = prop.get_family()[0]
    if family in families:
        commands.append(families[family])
    elif (any(font.name == family for font in fm.fontManager.ttflist)
          and rcParams["pgf.texsystem"] != "pdflatex"):
        commands.append(r"\setmainfont{%s}\rmfamily" % family)
    else:
        pass  # print warning?

    size = prop.get_size_in_points()
    commands.append(r"\fontsize{%f}{%f}" % (size, size * 1.2))

    styles = {"normal": r"", "italic": r"\itshape", "oblique": r"\slshape"}
    commands.append(styles[prop.get_style()])

    boldstyles = ["semibold", "demibold", "demi", "bold", "heavy",
                  "extra bold", "black"]
    if prop.get_weight() in boldstyles:
        commands.append(r"\bfseries")

    commands.append(r"\selectfont")
    return "".join(commands)


def make_pdf_to_png_converter():
    """
    Returns a function that converts a pdf file to a png file.
    """

    tools_available = []
    # check for pdftocairo
    try:
        subprocess.check_output(["pdftocairo", "-v"], stderr=subprocess.STDOUT)
        tools_available.append("pdftocairo")
    except OSError:
        pass
    # check for ghostscript
    gs, ver = mpl.checkdep_ghostscript()
    if gs:
        tools_available.append("gs")

    # pick converter
    if "pdftocairo" in tools_available:
        def cairo_convert(pdffile, pngfile, dpi):
            cmd = ["pdftocairo", "-singlefile", "-png", "-r", "%d" % dpi,
                   pdffile, os.path.splitext(pngfile)[0]]
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return cairo_convert
    elif "gs" in tools_available:
        def gs_convert(pdffile, pngfile, dpi):
            cmd = [gs,
                   '-dQUIET', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dNOPROMPT',
                   '-dUseCIEColor', '-dTextAlphaBits=4',
                   '-dGraphicsAlphaBits=4', '-dDOINTERPOLATE',
                   '-sDEVICE=png16m', '-sOutputFile=%s' % pngfile,
                   '-r%d' % dpi, pdffile]
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return gs_convert
    else:
        raise RuntimeError("No suitable pdf to png renderer found.")


class LatexError(Exception):
    def __init__(self, message, latex_output=""):
        super().__init__(message)
        self.latex_output = latex_output


class LatexManagerFactory:
    previous_instance = None

    @staticmethod
    def get_latex_manager():
        texcommand = rcParams["pgf.texsystem"]
        latex_header = LatexManager._build_latex_header()
        prev = LatexManagerFactory.previous_instance

        # Check if the previous instance of LatexManager can be reused.
        if (prev and prev.latex_header == latex_header
                and prev.texcommand == texcommand):
            _log.debug("reusing LatexManager")
            return prev
        else:
            _log.debug("creating LatexManager")
            new_inst = LatexManager()
            LatexManagerFactory.previous_instance = new_inst
            return new_inst


class LatexManager:
    """
    The LatexManager opens an instance of the LaTeX application for
    determining the metrics of text elements. The LaTeX environment can be
    modified by setting fonts and/or a custem preamble in the rc parameters.
    """
    _unclean_instances = weakref.WeakSet()

    @staticmethod
    def _build_latex_header():
        latex_preamble = get_preamble()
        latex_fontspec = get_fontspec()
        # Create LaTeX header with some content, else LaTeX will load some math
        # fonts later when we don't expect the additional output on stdout.
        # TODO: is this sufficient?
        latex_header = [r"\documentclass{minimal}",
                        latex_preamble,
                        latex_fontspec,
                        r"\begin{document}",
                        r"text $math \mu$",  # force latex to load fonts now
                        r"\typeout{pgf_backend_query_start}"]
        return "\n".join(latex_header)

    @staticmethod
    def _cleanup_remaining_instances():
        unclean_instances = list(LatexManager._unclean_instances)
        for latex_manager in unclean_instances:
            latex_manager._cleanup()

    def _stdin_writeln(self, s):
        self.latex_stdin_utf8.write(s)
        self.latex_stdin_utf8.write("\n")
        self.latex_stdin_utf8.flush()

    def _expect(self, s):
        exp = s.encode("utf8")
        buf = bytearray()
        while True:
            b = self.latex.stdout.read(1)
            buf += b
            if buf[-len(exp):] == exp:
                break
            if not len(b):
                raise LatexError("LaTeX process halted", buf.decode("utf8"))
        return buf.decode("utf8")

    def _expect_prompt(self):
        return self._expect("\n*")

    def __init__(self):
        # store references for __del__
        self._os_path = os.path
        self._shutil = shutil

        # create a tmp directory for running latex, remember to cleanup
        self.tmpdir = tempfile.mkdtemp(prefix="mpl_pgf_lm_")
        LatexManager._unclean_instances.add(self)

        # test the LaTeX setup to ensure a clean startup of the subprocess
        self.texcommand = rcParams["pgf.texsystem"]
        self.latex_header = LatexManager._build_latex_header()
        latex_end = "\n\\makeatletter\n\\@@end\n"
        try:
            latex = subprocess.Popen([self.texcommand, "-halt-on-error"],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     cwd=self.tmpdir)
        except FileNotFoundError:
            raise RuntimeError(
                "Latex command not found. Install %r or change "
                "pgf.texsystem to the desired command." % self.texcommand)
        except OSError:
            raise RuntimeError("Error starting process %r" % self.texcommand)
        test_input = self.latex_header + latex_end
        stdout, stderr = latex.communicate(test_input.encode("utf-8"))
        if latex.returncode != 0:
            raise LatexError("LaTeX returned an error, probably missing font "
                             "or error in preamble:\n%s" % stdout)

        # open LaTeX process for real work
        latex = subprocess.Popen([self.texcommand, "-halt-on-error"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 cwd=self.tmpdir)
        self.latex = latex
        self.latex_stdin_utf8 = codecs.getwriter("utf8")(self.latex.stdin)
        # write header with 'pgf_backend_query_start' token
        self._stdin_writeln(self._build_latex_header())
        # read all lines until our 'pgf_backend_query_start' token appears
        self._expect("*pgf_backend_query_start")
        self._expect_prompt()

        # cache for strings already processed
        self.str_cache = {}

    def _cleanup(self):
        if not self._os_path.isdir(self.tmpdir):
            return
        try:
            self.latex.communicate()
            self.latex_stdin_utf8.close()
            self.latex.stdout.close()
        except Exception:
            pass
        try:
            self._shutil.rmtree(self.tmpdir)
            LatexManager._unclean_instances.discard(self)
        except Exception:
            sys.stderr.write("error deleting tmp directory %s\n" % self.tmpdir)

    def __del__(self):
        _log.debug("deleting LatexManager")
        self._cleanup()

    def get_width_height_descent(self, text, prop):
        """
        Get the width, total height and descent for a text typesetted by the
        current LaTeX environment.
        """

        # apply font properties and define textbox
        prop_cmds = _font_properties_str(prop)
        textbox = "\\sbox0{%s %s}" % (prop_cmds, text)

        # check cache
        if textbox in self.str_cache:
            return self.str_cache[textbox]

        # send textbox to LaTeX and wait for prompt
        self._stdin_writeln(textbox)
        try:
            self._expect_prompt()
        except LatexError as e:
            raise ValueError("Error processing '{}'\nLaTeX Output:\n{}"
                             .format(text, e.latex_output))

        # typeout width, height and text offset of the last textbox
        self._stdin_writeln(r"\typeout{\the\wd0,\the\ht0,\the\dp0}")
        # read answer from latex and advance to the next prompt
        try:
            answer = self._expect_prompt()
        except LatexError as e:
            raise ValueError("Error processing '{}'\nLaTeX Output:\n{}"
                             .format(text, e.latex_output))

        # parse metrics from the answer string
        try:
            width, height, offset = answer.splitlines()[0].split(",")
        except:
            raise ValueError("Error processing '{}'\nLaTeX Output:\n{}"
                             .format(text, answer))
        w, h, o = float(width[:-2]), float(height[:-2]), float(offset[:-2])

        # the height returned from LaTeX goes from base to top.
        # the height matplotlib expects goes from bottom to top.
        self.str_cache[textbox] = (w, h + o, o)
        return w, h + o, o


class RendererPgf(RendererBase):

    def __init__(self, figure, fh, dummy=False):
        """
        Creates a new PGF renderer that translates any drawing instruction
        into text commands to be interpreted in a latex pgfpicture environment.

        Attributes
        ----------
        figure : `matplotlib.figure.Figure`
            Matplotlib figure to initialize height, width and dpi from.
        fh : file-like
            File handle for the output of the drawing commands.

        """
        RendererBase.__init__(self)
        self.dpi = figure.dpi
        self.fh = fh
        self.figure = figure
        self.image_counter = 0

        # get LatexManager instance
        self.latexManager = LatexManagerFactory.get_latex_manager()

        if dummy:
            # dummy==True deactivate all methods
            nop = lambda *args, **kwargs: None
            for m in RendererPgf.__dict__:
                if m.startswith("draw_"):
                    self.__dict__[m] = nop
        else:
            # if fh does not belong to a filename, deactivate draw_image
            if not hasattr(fh, 'name') or not os.path.exists(fh.name):
                warnings.warn("streamed pgf-code does not support raster "
                              "graphics, consider using the pgf-to-pdf option",
                              UserWarning, stacklevel=2)
                self.__dict__["draw_image"] = lambda *args, **kwargs: None

    def draw_markers(self, gc, marker_path, marker_trans, path, trans,
                     rgbFace=None):
        writeln(self.fh, r"\begin{pgfscope}")

        # convert from display units to in
        f = 1. / self.dpi

        # set style and clip
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)

        # build marker definition
        bl, tr = marker_path.get_extents(marker_trans).get_points()
        coords = bl[0] * f, bl[1] * f, tr[0] * f, tr[1] * f
        writeln(self.fh,
                r"\pgfsys@defobject{currentmarker}"
                r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}{" % coords)
        self._print_pgf_path(None, marker_path, marker_trans)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0,
                            fill=rgbFace is not None)
        writeln(self.fh, r"}")

        # draw marker for each vertex
        for point, code in path.iter_segments(trans, simplify=False):
            x, y = point[0] * f, point[1] * f
            writeln(self.fh, r"\begin{pgfscope}")
            writeln(self.fh, r"\pgfsys@transformshift{%fin}{%fin}" % (x, y))
            writeln(self.fh, r"\pgfsys@useobject{currentmarker}{}")
            writeln(self.fh, r"\end{pgfscope}")

        writeln(self.fh, r"\end{pgfscope}")

    def draw_path(self, gc, path, transform, rgbFace=None):
        writeln(self.fh, r"\begin{pgfscope}")
        # draw the path
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)
        self._print_pgf_path(gc, path, transform, rgbFace)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0,
                            fill=rgbFace is not None)
        writeln(self.fh, r"\end{pgfscope}")

        # if present, draw pattern on top
        if gc.get_hatch():
            writeln(self.fh, r"\begin{pgfscope}")
            self._print_pgf_path_styles(gc, rgbFace)

            # combine clip and path for clipping
            self._print_pgf_clip(gc)
            self._print_pgf_path(gc, path, transform, rgbFace)
            writeln(self.fh, r"\pgfusepath{clip}")

            # build pattern definition
            writeln(self.fh,
                    r"\pgfsys@defobject{currentpattern}"
                    r"{\pgfqpoint{0in}{0in}}{\pgfqpoint{1in}{1in}}{")
            writeln(self.fh, r"\begin{pgfscope}")
            writeln(self.fh,
                    r"\pgfpathrectangle"
                    r"{\pgfqpoint{0in}{0in}}{\pgfqpoint{1in}{1in}}")
            writeln(self.fh, r"\pgfusepath{clip}")
            scale = mpl.transforms.Affine2D().scale(self.dpi)
            self._print_pgf_path(None, gc.get_hatch_path(), scale)
            self._pgf_path_draw(stroke=True)
            writeln(self.fh, r"\end{pgfscope}")
            writeln(self.fh, r"}")
            # repeat pattern, filling the bounding rect of the path
            f = 1. / self.dpi
            (xmin, ymin), (xmax, ymax) = \
                path.get_extents(transform).get_points()
            xmin, xmax = f * xmin, f * xmax
            ymin, ymax = f * ymin, f * ymax
            repx, repy = int(math.ceil(xmax-xmin)), int(math.ceil(ymax-ymin))
            writeln(self.fh,
                    r"\pgfsys@transformshift{%fin}{%fin}" % (xmin, ymin))
            for iy in range(repy):
                for ix in range(repx):
                    writeln(self.fh, r"\pgfsys@useobject{currentpattern}{}")
                    writeln(self.fh, r"\pgfsys@transformshift{1in}{0in}")
                writeln(self.fh, r"\pgfsys@transformshift{-%din}{0in}" % repx)
                writeln(self.fh, r"\pgfsys@transformshift{0in}{1in}")

            writeln(self.fh, r"\end{pgfscope}")

    def _print_pgf_clip(self, gc):
        f = 1. / self.dpi
        # check for clip box
        bbox = gc.get_clip_rectangle()
        if bbox:
            p1, p2 = bbox.get_points()
            w, h = p2 - p1
            coords = p1[0] * f, p1[1] * f, w * f, h * f
            writeln(self.fh,
                    r"\pgfpathrectangle"
                    r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}"
                    % coords)
            writeln(self.fh, r"\pgfusepath{clip}")

        # check for clip path
        clippath, clippath_trans = gc.get_clip_path()
        if clippath is not None:
            self._print_pgf_path(gc, clippath, clippath_trans)
            writeln(self.fh, r"\pgfusepath{clip}")

    def _print_pgf_path_styles(self, gc, rgbFace):
        # cap style
        capstyles = {"butt": r"\pgfsetbuttcap",
                     "round": r"\pgfsetroundcap",
                     "projecting": r"\pgfsetrectcap"}
        writeln(self.fh, capstyles[gc.get_capstyle()])

        # join style
        joinstyles = {"miter": r"\pgfsetmiterjoin",
                      "round": r"\pgfsetroundjoin",
                      "bevel": r"\pgfsetbeveljoin"}
        writeln(self.fh, joinstyles[gc.get_joinstyle()])

        # filling
        has_fill = rgbFace is not None

        if gc.get_forced_alpha():
            fillopacity = strokeopacity = gc.get_alpha()
        else:
            strokeopacity = gc.get_rgb()[3]
            fillopacity = rgbFace[3] if has_fill and len(rgbFace) > 3 else 1.0

        if has_fill:
            writeln(self.fh,
                    r"\definecolor{currentfill}{rgb}{%f,%f,%f}"
                    % tuple(rgbFace[:3]))
            writeln(self.fh, r"\pgfsetfillcolor{currentfill}")
        if has_fill and fillopacity != 1.0:
            writeln(self.fh, r"\pgfsetfillopacity{%f}" % fillopacity)

        # linewidth and color
        lw = gc.get_linewidth() * mpl_pt_to_in * latex_in_to_pt
        stroke_rgba = gc.get_rgb()
        writeln(self.fh, r"\pgfsetlinewidth{%fpt}" % lw)
        writeln(self.fh,
                r"\definecolor{currentstroke}{rgb}{%f,%f,%f}"
                % stroke_rgba[:3])
        writeln(self.fh, r"\pgfsetstrokecolor{currentstroke}")
        if strokeopacity != 1.0:
            writeln(self.fh, r"\pgfsetstrokeopacity{%f}" % strokeopacity)

        # line style
        dash_offset, dash_list = gc.get_dashes()
        if dash_list is None:
            writeln(self.fh, r"\pgfsetdash{}{0pt}")
        else:
            writeln(self.fh,
                    r"\pgfsetdash{%s}{%fpt}"
                    % ("".join(r"{%fpt}" % dash for dash in dash_list),
                       dash_offset))

    def _print_pgf_path(self, gc, path, transform, rgbFace=None):
        f = 1. / self.dpi
        # check for clip box / ignore clip for filled paths
        bbox = gc.get_clip_rectangle() if gc else None
        if bbox and (rgbFace is None):
            p1, p2 = bbox.get_points()
            clip = (p1[0], p1[1], p2[0], p2[1])
        else:
            clip = None
        # build path
        for points, code in path.iter_segments(transform, clip=clip):
            if code == Path.MOVETO:
                x, y = tuple(points)
                writeln(self.fh,
                        r"\pgfpathmoveto{\pgfqpoint{%fin}{%fin}}" %
                        (f * x, f * y))
            elif code == Path.CLOSEPOLY:
                writeln(self.fh, r"\pgfpathclose")
            elif code == Path.LINETO:
                x, y = tuple(points)
                writeln(self.fh,
                        r"\pgfpathlineto{\pgfqpoint{%fin}{%fin}}" %
                        (f * x, f * y))
            elif code == Path.CURVE3:
                cx, cy, px, py = tuple(points)
                coords = cx * f, cy * f, px * f, py * f
                writeln(self.fh,
                        r"\pgfpathquadraticcurveto"
                        r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}"
                        % coords)
            elif code == Path.CURVE4:
                c1x, c1y, c2x, c2y, px, py = tuple(points)
                coords = c1x * f, c1y * f, c2x * f, c2y * f, px * f, py * f
                writeln(self.fh,
                        r"\pgfpathcurveto"
                        r"{\pgfqpoint{%fin}{%fin}}"
                        r"{\pgfqpoint{%fin}{%fin}}"
                        r"{\pgfqpoint{%fin}{%fin}}"
                        % coords)

    def _pgf_path_draw(self, stroke=True, fill=False):
        actions = []
        if stroke:
            actions.append("stroke")
        if fill:
            actions.append("fill")
        writeln(self.fh, r"\pgfusepath{%s}" % ",".join(actions))

    def option_scale_image(self):
        """
        pgf backend supports affine transform of image.
        """
        return True

    def option_image_nocomposite(self):
        """
        return whether to generate a composite image from multiple images on
        a set of axes
        """
        return not rcParams['image.composite_image']

    def draw_image(self, gc, x, y, im, transform=None):
        h, w = im.shape[:2]
        if w == 0 or h == 0:
            return

        # save the images to png files
        path = os.path.dirname(self.fh.name)
        fname = os.path.splitext(os.path.basename(self.fh.name))[0]
        fname_img = "%s-img%d.png" % (fname, self.image_counter)
        self.image_counter += 1
        _png.write_png(im[::-1], os.path.join(path, fname_img))

        # reference the image in the pgf picture
        writeln(self.fh, r"\begin{pgfscope}")
        self._print_pgf_clip(gc)
        f = 1. / self.dpi  # from display coords to inch
        if transform is None:
            writeln(self.fh,
                    r"\pgfsys@transformshift{%fin}{%fin}" % (x * f, y * f))
            w, h = w * f, h * f
        else:
            tr1, tr2, tr3, tr4, tr5, tr6 = transform.frozen().to_values()
            writeln(self.fh,
                    r"\pgfsys@transformcm{%f}{%f}{%f}{%f}{%fin}{%fin}" %
                    (tr1 * f, tr2 * f, tr3 * f, tr4 * f,
                     (tr5 + x) * f, (tr6 + y) * f))
            w = h = 1  # scale is already included in the transform
        interp = str(transform is None).lower()  # interpolation in PDF reader
        writeln(self.fh,
                r"\pgftext[left,bottom]"
                r"{\pgfimage[interpolate=%s,width=%fin,height=%fin]{%s}}" %
                (interp, w, h, fname_img))
        writeln(self.fh, r"\end{pgfscope}")

    def draw_tex(self, gc, x, y, s, prop, angle, ismath="TeX!", mtext=None):
        self.draw_text(gc, x, y, s, prop, angle, ismath, mtext)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # prepare string for tex
        s = common_texification(s)
        prop_cmds = _font_properties_str(prop)
        s = r"%s %s" % (prop_cmds, s)


        writeln(self.fh, r"\begin{pgfscope}")

        alpha = gc.get_alpha()
        if alpha != 1.0:
            writeln(self.fh, r"\pgfsetfillopacity{%f}" % alpha)
            writeln(self.fh, r"\pgfsetstrokeopacity{%f}" % alpha)
        rgb = tuple(gc.get_rgb())[:3]
        writeln(self.fh, r"\definecolor{textcolor}{rgb}{%f,%f,%f}" % rgb)
        writeln(self.fh, r"\pgfsetstrokecolor{textcolor}")
        writeln(self.fh, r"\pgfsetfillcolor{textcolor}")
        s = r"\color{textcolor}" + s

        f = 1.0 / self.figure.dpi
        text_args = []
        if mtext and (
                (angle == 0 or
                 mtext.get_rotation_mode() == "anchor") and
                mtext.get_va() != "center_baseline"):
            # if text anchoring can be supported, get the original coordinates
            # and add alignment information
            pos = mtext.get_unitless_position()
            x, y = mtext.get_transform().transform_point(pos)
            text_args.append("x=%fin" % (x * f))
            text_args.append("y=%fin" % (y * f))

            halign = {"left": "left", "right": "right", "center": ""}
            valign = {"top": "top", "bottom": "bottom",
                      "baseline": "base", "center": ""}
            text_args.append(halign[mtext.get_ha()])
            text_args.append(valign[mtext.get_va()])
        else:
            # if not, use the text layout provided by matplotlib
            text_args.append("x=%fin" % (x * f))
            text_args.append("y=%fin" % (y * f))
            text_args.append("left")
            text_args.append("base")

        if angle != 0:
            text_args.append("rotate=%f" % angle)

        writeln(self.fh, r"\pgftext[%s]{%s}" % (",".join(text_args), s))
        writeln(self.fh, r"\end{pgfscope}")

    def get_text_width_height_descent(self, s, prop, ismath):
        # check if the math is supposed to be displaystyled
        s = common_texification(s)

        # get text metrics in units of latex pt, convert to display units
        w, h, d = self.latexManager.get_width_height_descent(s, prop)
        # TODO: this should be latex_pt_to_in instead of mpl_pt_to_in
        # but having a little bit more space around the text looks better,
        # plus the bounding box reported by LaTeX is VERY narrow
        f = mpl_pt_to_in * self.dpi
        return w * f, h * f, d * f

    def flipy(self):
        return False

    def get_canvas_width_height(self):
        return self.figure.get_figwidth(), self.figure.get_figheight()

    def points_to_pixels(self, points):
        return points * mpl_pt_to_in * self.dpi

    def new_gc(self):
        return GraphicsContextPgf()


class GraphicsContextPgf(GraphicsContextBase):
    pass

########################################################################


class TmpDirCleaner:
    remaining_tmpdirs = set()

    @staticmethod
    def add(tmpdir):
        TmpDirCleaner.remaining_tmpdirs.add(tmpdir)

    @staticmethod
    def cleanup_remaining_tmpdirs():
        for tmpdir in TmpDirCleaner.remaining_tmpdirs:
            shutil.rmtree(
                tmpdir,
                onerror=lambda *args: print("error deleting tmp directory %s"
                                            % tmpdir, file=sys.stderr))


class FigureCanvasPgf(FigureCanvasBase):
    filetypes = {"pgf": "LaTeX PGF picture",
                 "pdf": "LaTeX compiled PGF picture",
                 "png": "Portable Network Graphics", }

    def get_default_filetype(self):
        return 'pdf'

    def _print_pgf_to_fh(self, fh, *args,
                         dryrun=False, bbox_inches_restore=None, **kwargs):
        if dryrun:
            renderer = RendererPgf(self.figure, None, dummy=True)
            self.figure.draw(renderer)
            return

        header_text = """%% Creator: Matplotlib, PGF backend
%%
%% To include the figure in your LaTeX document, write
%%   \\input{<filename>.pgf}
%%
%% Make sure the required packages are loaded in your preamble
%%   \\usepackage{pgf}
%%
%% Figures using additional raster images can only be included by \\input if
%% they are in the same directory as the main LaTeX file. For loading figures
%% from other directories you can use the `import` package
%%   \\usepackage{import}
%% and then include the figures with
%%   \\import{<path to file>}{<filename>.pgf}
%%
"""

        # append the preamble used by the backend as a comment for debugging
        header_info_preamble = ["%% Matplotlib used the following preamble"]
        for line in get_preamble().splitlines():
            header_info_preamble.append("%%   " + line)
        for line in get_fontspec().splitlines():
            header_info_preamble.append("%%   " + line)
        header_info_preamble.append("%%")
        header_info_preamble = "\n".join(header_info_preamble)

        # get figure size in inch
        w, h = self.figure.get_figwidth(), self.figure.get_figheight()
        dpi = self.figure.get_dpi()

        # create pgfpicture environment and write the pgf code
        fh.write(header_text)
        fh.write(header_info_preamble)
        fh.write("\n")
        writeln(fh, r"\begingroup")
        writeln(fh, r"\makeatletter")
        writeln(fh, r"\begin{pgfpicture}")
        writeln(fh,
                r"\pgfpathrectangle{\pgfpointorigin}{\pgfqpoint{%fin}{%fin}}"
                % (w, h))
        writeln(fh, r"\pgfusepath{use as bounding box, clip}")
        renderer = MixedModeRenderer(self.figure, w, h, dpi,
                                     RendererPgf(self.figure, fh),
                                     bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)

        # end the pgfpicture environment
        writeln(fh, r"\end{pgfpicture}")
        writeln(fh, r"\makeatother")
        writeln(fh, r"\endgroup")

    def print_pgf(self, fname_or_fh, *args, **kwargs):
        """
        Output pgf commands for drawing the figure so it can be included and
        rendered in latex documents.
        """
        if kwargs.get("dryrun", False):
            self._print_pgf_to_fh(None, *args, **kwargs)
            return

        # figure out where the pgf is to be written to
        if isinstance(fname_or_fh, str):
            with open(fname_or_fh, "w", encoding="utf-8") as fh:
                self._print_pgf_to_fh(fh, *args, **kwargs)
        elif is_writable_file_like(fname_or_fh):
            fh = codecs.getwriter("utf-8")(fname_or_fh)
            self._print_pgf_to_fh(fh, *args, **kwargs)
        else:
            raise ValueError("filename must be a path")

    def _print_pdf_to_fh(self, fh, *args, **kwargs):
        w, h = self.figure.get_figwidth(), self.figure.get_figheight()

        try:
            # create temporary directory for compiling the figure
            tmpdir = tempfile.mkdtemp(prefix="mpl_pgf_")
            fname_pgf = os.path.join(tmpdir, "figure.pgf")
            fname_tex = os.path.join(tmpdir, "figure.tex")
            fname_pdf = os.path.join(tmpdir, "figure.pdf")

            # print figure to pgf and compile it with latex
            self.print_pgf(fname_pgf, *args, **kwargs)

            latex_preamble = get_preamble()
            latex_fontspec = get_fontspec()
            latexcode = """
\\documentclass[12pt]{minimal}
\\usepackage[paperwidth=%fin, paperheight=%fin, margin=0in]{geometry}
%s
%s
\\usepackage{pgf}

\\begin{document}
\\centering
\\input{figure.pgf}
\\end{document}""" % (w, h, latex_preamble, latex_fontspec)
            pathlib.Path(fname_tex).write_text(latexcode, encoding="utf-8")

            texcommand = rcParams["pgf.texsystem"]
            cmdargs = [texcommand, "-interaction=nonstopmode",
                       "-halt-on-error", "figure.tex"]
            try:
                subprocess.check_output(
                    cmdargs, stderr=subprocess.STDOUT, cwd=tmpdir)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "%s was not able to process your file.\n\nFull log:\n%s"
                    % (texcommand, e.output))

            # copy file contents to target
            with open(fname_pdf, "rb") as fh_src:
                shutil.copyfileobj(fh_src, fh)
        finally:
            try:
                shutil.rmtree(tmpdir)
            except:
                TmpDirCleaner.add(tmpdir)

    def print_pdf(self, fname_or_fh, *args, **kwargs):
        """
        Use LaTeX to compile a Pgf generated figure to PDF.
        """
        if kwargs.get("dryrun", False):
            self._print_pgf_to_fh(None, *args, **kwargs)
            return

        # figure out where the pdf is to be written to
        if isinstance(fname_or_fh, str):
            with open(fname_or_fh, "wb") as fh:
                self._print_pdf_to_fh(fh, *args, **kwargs)
        elif is_writable_file_like(fname_or_fh):
            self._print_pdf_to_fh(fname_or_fh, *args, **kwargs)
        else:
            raise ValueError("filename must be a path or a file-like object")

    def _print_png_to_fh(self, fh, *args, **kwargs):
        converter = make_pdf_to_png_converter()

        try:
            # create temporary directory for pdf creation and png conversion
            tmpdir = tempfile.mkdtemp(prefix="mpl_pgf_")
            fname_pdf = os.path.join(tmpdir, "figure.pdf")
            fname_png = os.path.join(tmpdir, "figure.png")
            # create pdf and try to convert it to png
            self.print_pdf(fname_pdf, *args, **kwargs)
            converter(fname_pdf, fname_png, dpi=self.figure.dpi)
            # copy file contents to target
            with open(fname_png, "rb") as fh_src:
                shutil.copyfileobj(fh_src, fh)
        finally:
            try:
                shutil.rmtree(tmpdir)
            except:
                TmpDirCleaner.add(tmpdir)

    def print_png(self, fname_or_fh, *args, **kwargs):
        """
        Use LaTeX to compile a pgf figure to pdf and convert it to png.
        """
        if kwargs.get("dryrun", False):
            self._print_pgf_to_fh(None, *args, **kwargs)
            return

        if isinstance(fname_or_fh, str):
            with open(fname_or_fh, "wb") as fh:
                self._print_png_to_fh(fh, *args, **kwargs)
        elif is_writable_file_like(fname_or_fh):
            self._print_png_to_fh(fname_or_fh, *args, **kwargs)
        else:
            raise ValueError("filename must be a path or a file-like object")

    def get_renderer(self):
        return RendererPgf(self.figure, None, dummy=True)


class FigureManagerPgf(FigureManagerBase):
    pass


@_Backend.export
class _BackendPgf(_Backend):
    FigureCanvas = FigureCanvasPgf
    FigureManager = FigureManagerPgf


def _cleanup_all():
    LatexManager._cleanup_remaining_instances()
    TmpDirCleaner.cleanup_remaining_tmpdirs()


atexit.register(_cleanup_all)


class PdfPages:
    """
    A multi-page PDF file using the pgf backend

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> # Initialize:
    >>> with PdfPages('foo.pdf') as pdf:
    ...     # As many times as you like, create a figure fig and save it:
    ...     fig = plt.figure()
    ...     pdf.savefig(fig)
    ...     # When no figure is specified the current figure is saved
    ...     pdf.savefig()
    """
    __slots__ = (
        '_outputfile',
        'keep_empty',
        '_tmpdir',
        '_basename',
        '_fname_tex',
        '_fname_pdf',
        '_n_figures',
        '_file',
        'metadata',
    )

    def __init__(self, filename, *, keep_empty=True, metadata=None):
        """
        Create a new PdfPages object.

        Parameters
        ----------

        filename : str
            Plots using :meth:`PdfPages.savefig` will be written to a file at
            this location. Any older file with the same name is overwritten.
        keep_empty : bool, optional
            If set to False, then empty pdf files will be deleted automatically
            when closed.
        metadata : dictionary, optional
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            `{'Creator': 'My software', 'Author': 'Me',
            'Title': 'Awesome fig'}`

            The standard keys are `'Title'`, `'Author'`, `'Subject'`,
            `'Keywords'`, `'Producer'`, `'Creator'` and `'Trapped'`.
            Values have been predefined for `'Creator'` and `'Producer'`.
            They can be removed by setting them to the empty string.
        """
        self._outputfile = filename
        self._n_figures = 0
        self.keep_empty = keep_empty
        self.metadata = metadata or {}

        # create temporary directory for compiling the figure
        self._tmpdir = tempfile.mkdtemp(prefix="mpl_pgf_pdfpages_")
        self._basename = 'pdf_pages'
        self._fname_tex = os.path.join(self._tmpdir, self._basename + ".tex")
        self._fname_pdf = os.path.join(self._tmpdir, self._basename + ".pdf")
        self._file = open(self._fname_tex, 'wb')

    def _write_header(self, width_inches, height_inches):
        supported_keys = {
            'title', 'author', 'subject', 'keywords', 'creator',
            'producer', 'trapped'
        }
        infoDict = {
            'creator': 'matplotlib %s, https://matplotlib.org' % __version__,
            'producer': 'matplotlib pgf backend %s' % __version__,
        }
        metadata = {k.lower(): v for k, v in self.metadata.items()}
        infoDict.update(metadata)
        hyperref_options = ''
        for k, v in infoDict.items():
            if k not in supported_keys:
                raise ValueError(
                    'Not a supported pdf metadata field: "{}"'.format(k)
                )
            hyperref_options += 'pdf' + k + '={' + str(v) + '},'

        latex_preamble = get_preamble()
        latex_fontspec = get_fontspec()
        latex_header = r"""\PassOptionsToPackage{{
  {metadata}
}}{{hyperref}}
\RequirePackage{{hyperref}}
\documentclass[12pt]{{minimal}}
\usepackage[
    paperwidth={width}in,
    paperheight={height}in,
    margin=0in
]{{geometry}}
{preamble}
{fontspec}
\usepackage{{pgf}}
\setlength{{\parindent}}{{0pt}}

\begin{{document}}%%
""".format(
            width=width_inches,
            height=height_inches,
            preamble=latex_preamble,
            fontspec=latex_fontspec,
            metadata=hyperref_options,
        )
        self._file.write(latex_header.encode('utf-8'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Finalize this object, running LaTeX in a temporary directory
        and moving the final pdf file to `filename`.
        """
        self._file.write(rb'\end{document}\n')
        self._file.close()

        if self._n_figures > 0:
            try:
                self._run_latex()
            finally:
                try:
                    shutil.rmtree(self._tmpdir)
                except:
                    TmpDirCleaner.add(self._tmpdir)
        elif self.keep_empty:
            open(self._outputfile, 'wb').close()

    def _run_latex(self):
        texcommand = rcParams["pgf.texsystem"]
        cmdargs = [
            texcommand,
            "-interaction=nonstopmode",
            "-halt-on-error",
            os.path.basename(self._fname_tex),
        ]
        try:
            subprocess.check_output(
                cmdargs, stderr=subprocess.STDOUT, cwd=self._tmpdir
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "%s was not able to process your file.\n\nFull log:\n%s"
                % (texcommand, e.output.decode('utf-8')))

        # copy file contents to target
        shutil.copyfile(self._fname_pdf, self._outputfile)

    def savefig(self, figure=None, **kwargs):
        """
        Saves a :class:`~matplotlib.figure.Figure` to this file as a new page.

        Any other keyword arguments are passed to
        :meth:`~matplotlib.figure.Figure.savefig`.

        Parameters
        ----------

        figure : :class:`~matplotlib.figure.Figure` or int, optional
            Specifies what figure is saved to file. If not specified, the
            active figure is saved. If a :class:`~matplotlib.figure.Figure`
            instance is provided, this figure is saved. If an int is specified,
            the figure instance to save is looked up by number.
        """
        if not isinstance(figure, Figure):
            if figure is None:
                manager = Gcf.get_active()
            else:
                manager = Gcf.get_fig_manager(figure)
            if manager is None:
                raise ValueError("No figure {}".format(figure))
            figure = manager.canvas.figure

        try:
            orig_canvas = figure.canvas
            figure.canvas = FigureCanvasPgf(figure)

            width, height = figure.get_size_inches()
            if self._n_figures == 0:
                self._write_header(width, height)
            else:
                # \pdfpagewidth and \pdfpageheight exist on pdftex, xetex, and
                # luatex<0.85; they were renamed to \pagewidth and \pageheight
                # on luatex>=0.85.
                self._file.write(
                    br'\newpage'
                    br'\ifdefined\pdfpagewidth\pdfpagewidth'
                    br'\else\pagewidth\fi=%ain'
                    br'\ifdefined\pdfpageheight\pdfpageheight'
                    br'\else\pageheight\fi=%ain'
                    b'%%\n' % (width, height)
                )

            figure.savefig(self._file, format="pgf", **kwargs)
            self._n_figures += 1
        finally:
            figure.canvas = orig_canvas

    def get_pagecount(self):
        """
        Returns the current number of pages in the multipage pdf file.
        """
        return self._n_figures
