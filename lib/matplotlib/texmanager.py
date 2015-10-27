"""
This module supports embedded TeX expressions in matplotlib via dvipng
and dvips for the raster and postscript backends.  The tex and
dvipng/dvips information is cached in ~/.matplotlib/tex.cache for reuse between
sessions

Requirements:

* latex
* \*Agg backends: dvipng
* PS backend: latex w/ psfrag, dvips, and Ghostscript 8.51
  (older versions do not work properly)

Backends:

* \*Agg
* PS
* PDF

For raster output, you can get RGBA numpy arrays from TeX expressions
as follows::

  texmanager = TexManager()
  s = ('\\TeX\\ is Number '
       '$\\displaystyle\\sum_{n=1}^\\infty\\frac{-e^{i\pi}}{2^n}$!')
  Z = self.texmanager.get_rgba(s, size=12, dpi=80, rgb=(1,0,0))

To enable tex rendering of all text in your matplotlib figure, set
text.usetex in your matplotlibrc file or include these two lines in
your script::

  from matplotlib import rc
  rc('text', usetex=True)

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import copy
import glob
import os
import shutil
import sys
import warnings

from hashlib import md5

import distutils.version
import numpy as np
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib._png import read_png
from matplotlib.cbook import mkdirs
from matplotlib.compat.subprocess import Popen, PIPE, STDOUT
import matplotlib.dviread as dviread
import re

DEBUG = False

if sys.platform.startswith('win'):
    cmd_split = '&'
else:
    cmd_split = ';'


def dvipng_hack_alpha():
    try:
        p = Popen(['dvipng', '-version'], stdin=PIPE, stdout=PIPE,
                  stderr=STDOUT, close_fds=(sys.platform != 'win32'))
        stdout, stderr = p.communicate()
    except OSError:
        mpl.verbose.report('No dvipng was found', 'helpful')
        return False
    lines = stdout.decode(sys.getdefaultencoding()).split('\n')
    for line in lines:
        if line.startswith('dvipng '):
            version = line.split()[-1]
            mpl.verbose.report('Found dvipng version %s' % version,
                               'helpful')
            version = distutils.version.LooseVersion(version)
            return version < distutils.version.LooseVersion('1.6')
    mpl.verbose.report('Unexpected response from dvipng -version', 'helpful')
    return False


class TexManager(object):
    """
    Convert strings to dvi files using TeX, caching the results to a
    working dir
    """

    oldpath = mpl.get_home()
    if oldpath is None:
        oldpath = mpl.get_data_path()
    oldcache = os.path.join(oldpath, '.tex.cache')

    cachedir = mpl.get_cachedir()
    if cachedir is not None:
        texcache = os.path.join(cachedir, 'tex.cache')
    else:
        # Should only happen in a restricted environment (such as Google App
        # Engine). Deal with this gracefully by not creating a cache directory.
        texcache = None

    if os.path.exists(oldcache):
        if texcache is not None:
            try:
                shutil.move(oldcache, texcache)
            except IOError as e:
                warnings.warn('File could not be renamed: %s' % e)
            else:
                warnings.warn("""\
Found a TeX cache dir in the deprecated location "%s".
    Moving it to the new default location "%s".""" % (oldcache, texcache))
        else:
            warnings.warn("""\
Could not rename old TeX cache dir "%s": a suitable configuration
    directory could not be found.""" % oldcache)

    if texcache is not None:
        mkdirs(texcache)

    _dvipng_hack_alpha = None
    #_dvipng_hack_alpha = dvipng_hack_alpha()
    # mappable cache of
    rgba_arrayd = {}
    grey_arrayd = {}
    postscriptd = {}
    pscnt = 0

    serif = ('cmr', '')
    sans_serif = ('cmss', '')
    monospace = ('cmtt', '')
    cursive = ('pzc', '\\usepackage{chancery}')
    font_family = 'serif'
    font_families = ('serif', 'sans-serif', 'cursive', 'monospace')

    font_info = {'new century schoolbook': ('pnc',
                                            r'\renewcommand{\rmdefault}{pnc}'),
                 'bookman': ('pbk', r'\renewcommand{\rmdefault}{pbk}'),
                 'times': ('ptm', '\\usepackage{mathptmx}'),
                 'palatino': ('ppl', '\\usepackage{mathpazo}'),
                 'zapf chancery': ('pzc', '\\usepackage{chancery}'),
                 'cursive': ('pzc', '\\usepackage{chancery}'),
                 'charter': ('pch', '\\usepackage{charter}'),
                 'serif': ('cmr', ''),
                 'sans-serif': ('cmss', ''),
                 'helvetica': ('phv', '\\usepackage{helvet}'),
                 'avant garde': ('pag', '\\usepackage{avant}'),
                 'courier': ('pcr', '\\usepackage{courier}'),
                 'monospace': ('cmtt', ''),
                 'computer modern roman': ('cmr', ''),
                 'computer modern sans serif': ('cmss', ''),
                 'computer modern typewriter': ('cmtt', '')}

    _rc_cache = None
    _rc_cache_keys = (('text.latex.preamble', ) +
                      tuple(['font.' + n for n in ('family', ) +
                             font_families]))

    def __init__(self):

        if self.texcache is None:
            raise RuntimeError(
                ('Cannot create TexManager, as there is no cache directory '
                 'available'))

        mkdirs(self.texcache)
        ff = rcParams['font.family']
        if len(ff) == 1 and ff[0].lower() in self.font_families:
            self.font_family = ff[0].lower()
        elif isinstance(ff, six.string_types) and ff.lower() in self.font_families:
            self.font_family = ff.lower()
        else:
            mpl.verbose.report(
                'font.family must be one of (%s) when text.usetex is True. '
                'serif will be used by default.' %
                   ', '.join(self.font_families),
                'helpful')
            self.font_family = 'serif'

        fontconfig = [self.font_family]
        for font_family, font_family_attr in [(ff, ff.replace('-', '_'))
                                              for ff in self.font_families]:
            for font in rcParams['font.' + font_family]:
                if font.lower() in self.font_info:
                    setattr(self, font_family_attr,
                            self.font_info[font.lower()])
                    if DEBUG:
                        print('family: %s, font: %s, info: %s' %
                              (font_family, font,
                               self.font_info[font.lower()]))
                    break
                else:
                    if DEBUG:
                        print('$s font is not compatible with usetex')
            else:
                mpl.verbose.report('No LaTeX-compatible font found for the '
                                   '%s font family in rcParams. Using '
                                   'default.' % ff, 'helpful')
                setattr(self, font_family_attr, self.font_info[font_family])
            fontconfig.append(getattr(self, font_family_attr)[0])
        # Add a hash of the latex preamble to self._fontconfig so that the
        # correct png is selected for strings rendered with same font and dpi
        # even if the latex preamble changes within the session
        preamble_bytes = six.text_type(self.get_custom_preamble()).encode('utf-8')
        fontconfig.append(md5(preamble_bytes).hexdigest())
        self._fontconfig = ''.join(fontconfig)

        # The following packages and commands need to be included in the latex
        # file's preamble:
        cmd = [self.serif[1], self.sans_serif[1], self.monospace[1]]
        if self.font_family == 'cursive':
            cmd.append(self.cursive[1])
        while '\\usepackage{type1cm}' in cmd:
            cmd.remove('\\usepackage{type1cm}')
        cmd = '\n'.join(cmd)
        self._font_preamble = '\n'.join(['\\usepackage{type1cm}', cmd,
                                         '\\usepackage{textcomp}'])

    def get_basefile(self, tex, fontsize, dpi=None):
        """
        returns a filename based on a hash of the string, fontsize, and dpi
        """
        s = ''.join([tex, self.get_font_config(), '%f' % fontsize,
                     self.get_custom_preamble(), str(dpi or '')])
        # make sure hash is consistent for all strings, regardless of encoding:
        bytes = six.text_type(s).encode('utf-8')
        return os.path.join(self.texcache, md5(bytes).hexdigest())

    def get_font_config(self):
        """Reinitializes self if relevant rcParams on have changed."""
        if self._rc_cache is None:
            self._rc_cache = dict([(k, None) for k in self._rc_cache_keys])
        changed = [par for par in self._rc_cache_keys
                   if rcParams[par] != self._rc_cache[par]]
        if changed:
            if DEBUG:
                print('DEBUG following keys changed:', changed)
            for k in changed:
                if DEBUG:
                    print('DEBUG %-20s: %-10s -> %-10s' %
                          (k, self._rc_cache[k], rcParams[k]))
                # deepcopy may not be necessary, but feels more future-proof
                self._rc_cache[k] = copy.deepcopy(rcParams[k])
            if DEBUG:
                print('DEBUG RE-INIT\nold fontconfig:', self._fontconfig)
            self.__init__()
        if DEBUG:
            print('DEBUG fontconfig:', self._fontconfig)
        return self._fontconfig

    def get_font_preamble(self):
        """
        returns a string containing font configuration for the tex preamble
        """
        return self._font_preamble

    def get_custom_preamble(self):
        """returns a string containing user additions to the tex preamble"""
        return '\n'.join(rcParams['text.latex.preamble'])

    def _get_shell_cmd(self, *args):
        """
        On windows, changing directories can be complicated by the presence of
        multiple drives. get_shell_cmd deals with this issue.
        """
        if sys.platform == 'win32':
            command = ['%s' % os.path.splitdrive(self.texcache)[0]]
        else:
            command = []
        command.extend(args)
        return ' && '.join(command)

    def make_tex(self, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific font size

        returns the file name
        """
        basefile = self.get_basefile(tex, fontsize)
        texfile = '%s.tex' % basefile
        custom_preamble = self.get_custom_preamble()
        fontcmd = {'sans-serif': r'{\sffamily %s}',
                   'monospace': r'{\ttfamily %s}'}.get(self.font_family,
                                                       r'{\rmfamily %s}')
        tex = fontcmd % tex

        if rcParams['text.latex.unicode']:
            unicode_preamble = """\\usepackage{ucs}
\\usepackage[utf8x]{inputenc}"""
        else:
            unicode_preamble = ''

        s = """\\documentclass{article}
%s
%s
%s
\\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}
\\pagestyle{empty}
\\begin{document}
\\fontsize{%f}{%f}%s
\\end{document}
""" % (self._font_preamble, unicode_preamble, custom_preamble,
       fontsize, fontsize * 1.25, tex)
        with open(texfile, 'wb') as fh:
            if rcParams['text.latex.unicode']:
                fh.write(s.encode('utf8'))
            else:
                try:
                    fh.write(s.encode('ascii'))
                except UnicodeEncodeError as err:
                    mpl.verbose.report("You are using unicode and latex, but "
                                       "have not enabled the matplotlib "
                                       "'text.latex.unicode' rcParam.",
                                       'helpful')
                    raise

        return texfile

    _re_vbox = re.compile(
        r"MatplotlibBox:\(([\d.]+)pt\+([\d.]+)pt\)x([\d.]+)pt")

    def make_tex_preview(self, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific
        font size.  It uses the preview.sty to determin the dimension
        (width, height, descent) of the output.

        returns the file name
        """
        basefile = self.get_basefile(tex, fontsize)
        texfile = '%s.tex' % basefile
        custom_preamble = self.get_custom_preamble()
        fontcmd = {'sans-serif': r'{\sffamily %s}',
                   'monospace': r'{\ttfamily %s}'}.get(self.font_family,
                                                       r'{\rmfamily %s}')
        tex = fontcmd % tex

        if rcParams['text.latex.unicode']:
            unicode_preamble = """\\usepackage{ucs}
\\usepackage[utf8x]{inputenc}"""
        else:
            unicode_preamble = ''

        # newbox, setbox, immediate, etc. are used to find the box
        # extent of the rendered text.

        s = """\\documentclass{article}
%s
%s
%s
\\usepackage[active,showbox,tightpage]{preview}
\\usepackage[papersize={72in,72in},body={70in,70in},margin={1in,1in}]{geometry}

%% we override the default showbox as it is treated as an error and makes
%% the exit status not zero
\\def\\showbox#1{\\immediate\\write16{MatplotlibBox:(\\the\\ht#1+\\the\\dp#1)x\\the\\wd#1}}

\\begin{document}
\\begin{preview}
{\\fontsize{%f}{%f}%s}
\\end{preview}
\\end{document}
""" % (self._font_preamble, unicode_preamble, custom_preamble,
       fontsize, fontsize * 1.25, tex)
        with open(texfile, 'wb') as fh:
            if rcParams['text.latex.unicode']:
                fh.write(s.encode('utf8'))
            else:
                try:
                    fh.write(s.encode('ascii'))
                except UnicodeEncodeError as err:
                    mpl.verbose.report("You are using unicode and latex, but "
                                       "have not enabled the matplotlib "
                                       "'text.latex.unicode' rcParam.",
                                       'helpful')
                    raise

        return texfile

    def make_dvi(self, tex, fontsize):
        """
        generates a dvi file containing latex's layout of tex string

        returns the file name
        """

        if rcParams['text.latex.preview']:
            return self.make_dvi_preview(tex, fontsize)

        basefile = self.get_basefile(tex, fontsize)
        dvifile = '%s.dvi' % basefile

        if DEBUG or not os.path.exists(dvifile):
            texfile = self.make_tex(tex, fontsize)
            outfile = basefile + '.output'
            command = self._get_shell_cmd(
                'cd "%s"' % self.texcache,
                'latex -interaction=nonstopmode %s > "%s"' %
                (os.path.split(texfile)[-1], outfile))
            mpl.verbose.report(command, 'debug')
            exit_status = os.system(command)
            try:
                with open(outfile) as fh:
                    report = fh.read()
            except IOError:
                report = 'No latex error report available.'
            try:
                os.stat(dvifile)
                exists = True
            except OSError:
                exists = False
            if exit_status or not exists:
                raise RuntimeError(
                    ('LaTeX was not able to process the following '
                     'string:\n%s\nHere is the full report generated by '
                     'LaTeX: \n\n' % repr(tex.encode('unicode_escape')) +
                     report))
            else:
                mpl.verbose.report(report, 'debug')
            for fname in glob.glob(basefile + '*'):
                if fname.endswith('dvi'):
                    pass
                elif fname.endswith('tex'):
                    pass
                else:
                    try:
                        os.remove(fname)
                    except OSError:
                        pass

        return dvifile

    def make_dvi_preview(self, tex, fontsize):
        """
        generates a dvi file containing latex's layout of tex
        string. It calls make_tex_preview() method and store the size
        information (width, height, descent) in a separte file.

        returns the file name
        """
        basefile = self.get_basefile(tex, fontsize)
        dvifile = '%s.dvi' % basefile
        baselinefile = '%s.baseline' % basefile

        if (DEBUG or not os.path.exists(dvifile) or
                not os.path.exists(baselinefile)):
            texfile = self.make_tex_preview(tex, fontsize)
            outfile = basefile + '.output'
            command = self._get_shell_cmd(
                'cd "%s"' % self.texcache,
                'latex -interaction=nonstopmode %s > "%s"' %
                (os.path.split(texfile)[-1], outfile))
            mpl.verbose.report(command, 'debug')
            exit_status = os.system(command)
            try:
                with open(outfile) as fh:
                    report = fh.read()

            except IOError:
                report = 'No latex error report available.'
            if exit_status:
                raise RuntimeError(
                    ('LaTeX was not able to process the following '
                     'string:\n%s\nHere is the full report generated by '
                     'LaTeX: \n\n' % repr(tex)) + report)
            else:
                mpl.verbose.report(report, 'debug')

            # find the box extent information in the latex output
            # file and store them in ".baseline" file
            m = TexManager._re_vbox.search(report)
            with open(basefile + '.baseline', "w") as fh:
                fh.write(" ".join(m.groups()))

            for fname in glob.glob(basefile + '*'):
                if fname.endswith('dvi'):
                    pass
                elif fname.endswith('tex'):
                    pass
                elif fname.endswith('baseline'):
                    pass
                else:
                    try:
                        os.remove(fname)
                    except OSError:
                        pass

        return dvifile

    def make_png(self, tex, fontsize, dpi):
        """
        generates a png file containing latex's rendering of tex string

        returns the filename
        """
        basefile = self.get_basefile(tex, fontsize, dpi)
        pngfile = '%s.png' % basefile

        # see get_rgba for a discussion of the background
        if DEBUG or not os.path.exists(pngfile):
            dvifile = self.make_dvi(tex, fontsize)
            outfile = basefile + '.output'
            command = self._get_shell_cmd(
                'cd "%s"' % self.texcache,
                'dvipng -bg Transparent -D %s -T tight -o "%s" "%s" > "%s"' %
                (dpi, os.path.split(pngfile)[-1],
                os.path.split(dvifile)[-1], outfile))
            mpl.verbose.report(command, 'debug')
            exit_status = os.system(command)
            try:
                with open(outfile) as fh:
                    report = fh.read()
            except IOError:
                report = 'No dvipng error report available.'
            if exit_status:
                raise RuntimeError(
                    'dvipng was not able to process the following '
                    'file:\n%s\nHere is the full report generated by '
                    'dvipng: \n\n' % dvifile + report)
            else:
                mpl.verbose.report(report, 'debug')
            try:
                os.remove(outfile)
            except OSError:
                pass

        return pngfile

    def make_ps(self, tex, fontsize):
        """
        generates a postscript file containing latex's rendering of tex string

        returns the file name
        """
        basefile = self.get_basefile(tex, fontsize)
        psfile = '%s.epsf' % basefile

        if DEBUG or not os.path.exists(psfile):
            dvifile = self.make_dvi(tex, fontsize)
            outfile = basefile + '.output'
            command = self._get_shell_cmd(
                'cd "%s"' % self.texcache,
                'dvips -q -E -o "%s" "%s" > "%s"' %
                (os.path.split(psfile)[-1],
                 os.path.split(dvifile)[-1], outfile))
            mpl.verbose.report(command, 'debug')
            exit_status = os.system(command)
            with open(outfile) as fh:
                if exit_status:
                    raise RuntimeError(
                        'dvipng was not able to process the flowing '
                        'file:\n%s\nHere is the full report generated by '
                        'dvipng: \n\n' % dvifile + fh.read())
                else:
                    mpl.verbose.report(fh.read(), 'debug')
            os.remove(outfile)

        return psfile

    def get_ps_bbox(self, tex, fontsize):
        """
        returns a list containing the postscript bounding box for latex's
        rendering of the tex string
        """
        psfile = self.make_ps(tex, fontsize)
        with open(psfile) as ps:
            for line in ps:
                if line.startswith('%%BoundingBox:'):
                    return [int(val) for val in line.split()[1:]]
        raise RuntimeError('Could not parse %s' % psfile)

    def get_grey(self, tex, fontsize=None, dpi=None):
        """returns the alpha channel"""
        key = tex, self.get_font_config(), fontsize, dpi
        alpha = self.grey_arrayd.get(key)

        if alpha is None:
            pngfile = self.make_png(tex, fontsize, dpi)
            X = read_png(os.path.join(self.texcache, pngfile))

            if rcParams['text.dvipnghack'] is not None:
                hack = rcParams['text.dvipnghack']
            else:
                if TexManager._dvipng_hack_alpha is None:
                    TexManager._dvipng_hack_alpha = dvipng_hack_alpha()
                hack = TexManager._dvipng_hack_alpha

            if hack:
                # hack the alpha channel
                # dvipng assumed a constant background, whereas we want to
                # overlay these rasters with antialiasing over arbitrary
                # backgrounds that may have other figure elements under them.
                # When you set dvipng -bg Transparent, it actually makes the
                # alpha channel 1 and does the background compositing and
                # antialiasing itself and puts the blended data in the rgb
                # channels.  So what we do is extract the alpha information
                # from the red channel, which is a blend of the default dvipng
                # background (white) and foreground (black).  So the amount of
                # red (or green or blue for that matter since white and black
                # blend to a grayscale) is the alpha intensity.  Once we
                # extract the correct alpha information, we assign it to the
                # alpha channel properly and let the users pick their rgb.  In
                # this way, we can overlay tex strings on arbitrary
                # backgrounds with antialiasing
                #
                # red = alpha*red_foreground + (1-alpha)*red_background
                #
                # Since the foreground is black (0) and the background is
                # white (1) this reduces to red = 1-alpha or alpha = 1-red
                #alpha = npy.sqrt(1-X[:,:,0]) # should this be sqrt here?
                alpha = 1 - X[:, :, 0]
            else:
                alpha = X[:, :, -1]

            self.grey_arrayd[key] = alpha
        return alpha

    def get_rgba(self, tex, fontsize=None, dpi=None, rgb=(0, 0, 0)):
        """
        Returns latex's rendering of the tex string as an rgba array
        """
        if not fontsize:
            fontsize = rcParams['font.size']
        if not dpi:
            dpi = rcParams['savefig.dpi']
        r, g, b = rgb
        key = tex, self.get_font_config(), fontsize, dpi, tuple(rgb)
        Z = self.rgba_arrayd.get(key)

        if Z is None:
            alpha = self.get_grey(tex, fontsize, dpi)

            Z = np.zeros((alpha.shape[0], alpha.shape[1], 4), np.float)

            Z[:, :, 0] = r
            Z[:, :, 1] = g
            Z[:, :, 2] = b
            Z[:, :, 3] = alpha
            self.rgba_arrayd[key] = Z

        return Z

    def get_text_width_height_descent(self, tex, fontsize, renderer=None):
        """
        return width, heigth and descent of the text.
        """
        if tex.strip() == '':
            return 0, 0, 0

        if renderer:
            dpi_fraction = renderer.points_to_pixels(1.)
        else:
            dpi_fraction = 1.

        if rcParams['text.latex.preview']:
            # use preview.sty
            basefile = self.get_basefile(tex, fontsize)
            baselinefile = '%s.baseline' % basefile

            if DEBUG or not os.path.exists(baselinefile):
                dvifile = self.make_dvi_preview(tex, fontsize)

            with open(baselinefile) as fh:
                l = fh.read().split()
            height, depth, width = [float(l1) * dpi_fraction for l1 in l]
            return width, height + depth, depth

        else:
            # use dviread. It sometimes returns a wrong descent.
            dvifile = self.make_dvi(tex, fontsize)
            dvi = dviread.Dvi(dvifile, 72 * dpi_fraction)
            try:
                page = next(iter(dvi))
            finally:
                dvi.close()
            # A total height (including the descent) needs to be returned.
            return page.width, page.height + page.descent, page.descent
