"""
This module supports embedded TeX expressions in matplotlib via dvipng
and dvips for the raster and postscript backends.  The tex and
dvipng/dvips information is cached in ~/.matplotlib/tex.cache for reuse between
sessions

Requirements:

  tex

  *Agg backends: dvipng

  PS backend: latex w/ psfrag, dvips, and Ghostscript 8.51
  (older versions do not work properly)

Backends:

  Only supported on *Agg and PS backends currently
  

For raster output, you can get RGBA numerix arrays from TeX expressions
as follows

  texmanager = TexManager()
  s = r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!'
  Z = self.texmanager.get_rgba(s, size=12, dpi=80, rgb=(1,0,0))

To enable tex rendering of all text in your matplotlib figure, set
text.usetex in your matplotlibrc file (http://matplotlib.sf.net/matplotlibrc)
or include these two lines in your script:
from matplotlib import rc
rc('text', usetex=True)

"""

import glob, md5, os, shutil, sys, warnings
from tempfile import gettempdir
from matplotlib import get_configdir, get_home, get_data_path, \
     rcParams, verbose
from matplotlib._image import readpng
from matplotlib.numerix import ravel, where, array, \
     zeros, Float, absolute, nonzero, sqrt
     
debug = False

if sys.platform.startswith('win'): cmd_split = '&'
else: cmd_split = ';'


def get_dvipng_version():
    stdin, stdout = os.popen4('dvipng --version')
    for line in stdout:
        if line.startswith('dvipng '):
            version = line.split()[-1]
            verbose.report('Found dvipng version %s'% version, 
                'helpful')
            return version
    raise RuntimeError('Could not obtain dvipng version')


class TexManager:
    """
    Convert strings to dvi files using TeX, caching the results to a
    working dir
    """
    
    oldpath = get_home()
    if oldpath is None: oldpath = get_data_path()
    oldcache = os.path.join(oldpath, '.tex.cache')

    configdir = get_configdir()
    texcache = os.path.join(configdir, 'tex.cache')
    

    if os.path.exists(oldcache):
        print >> sys.stderr, """\
WARNING: found a TeX cache dir in the deprecated location "%s".
  Moving it to the new default location "%s"."""%(oldcache, texcache)
        shutil.move(oldcache, texcache)
    if not os.path.exists(texcache):
        os.mkdir(texcache)

    dvipngVersion = get_dvipng_version()

    arrayd = {}
    postscriptd = {}
    pscnt = 0
    
    serif = ('cmr', '')
    sans_serif = ('cmss', '')
    monospace = ('cmtt', '')
    cursive = ('pzc', r'\usepackage{chancery}')
    font_family = 'serif'
    
    font_info = {'new century schoolbook': ('pnc', r'\renewcommand{\rmdefault}{pnc}'),
                'bookman': ('pbk', r'\renewcommand{\rmdefault}{pbk}'),
                'times': ('ptm', r'\usepackage{mathptmx}'),
                'palatino': ('ppl', r'\usepackage{mathpazo}'),
                'zapf chancery': ('pzc', r'\usepackage{chancery}'),
                'charter': ('pch', r'\usepackage{charter}'),
                'serif': ('cmr', ''),
                'sans-serif': ('cmss', ''),
                'helvetica': ('phv', r'\usepackage{helvet}'),
                'avant garde': ('pag', r'\usepackage{avant}'),
                'courier': ('pcr', r'\usepackage{courier}'),
                'monospace': ('cmtt', ''),
                'computer modern roman': ('cmr', ''),
                'computer modern sans serif': ('cmss', ''),
                'computer modern typewriter': ('cmtt', '')}

    def __init__(self):
        
        if not os.path.isdir(self.texcache):
            os.mkdir(self.texcache)
        if rcParams['font.family'].lower() in ('serif', 'sans-serif', 'cursive', 'monospace'):
            self.font_family = rcParams['font.family'].lower()
        else:
            warnings.warn('The %s font family is not compatible with LaTeX. serif will be used by default.' % ff)
            self.font_family = 'serif'
        self._fontconfig = self.font_family
        for font in rcParams['font.serif']:
            try:
                self.serif = self.font_info[font.lower()]
            except KeyError:
                continue
            else:
                break
        self._fontconfig += self.serif[0]
        for font in rcParams['font.sans-serif']:
            try:
                self.sans_serif = self.font_info[font.lower()]
            except KeyError:
                continue
            else:
                break
        self._fontconfig += self.sans_serif[0]
        for font in rcParams['font.monospace']:
            try:
                self.monospace = self.font_info[font.lower()]
            except KeyError:
                continue
            else:
                break                
        self._fontconfig += self.monospace[0]
        for font in rcParams['font.cursive']:
            try:
                self.cursive = self.font_info[font.lower()]
            except KeyError:
                continue
            else:
                break                
        self._fontconfig += self.cursive[0]
        
        # The following packages and commands need to be included in the latex 
        # file's preamble:
        cmd = [self.serif[1], self.sans_serif[1], self.monospace[1]]
        if self.font_family == 'cursive': cmd.append(self.cursive[1])
        while r'\usepackage{type1cm}' in cmd:
            cmd.remove(r'\usepackage{type1cm}')
        cmd = '\n'.join(cmd)
        self._font_preamble = '\n'.join([r'\usepackage{type1cm}',
                             cmd,
                             r'\usepackage{textcomp}'])
        
    def get_basefile(self, tex, fontsize, dpi=None):
        s = tex + self._fontconfig + ('%f'%fontsize)
        if dpi: s += ('%s'%dpi)
        return os.path.join(self.texcache, md5.md5(s).hexdigest())
        
    def get_font_config(self):
        return self._fontconfig
        
    def get_font_preamble(self):
        return self._font_preamble
        
    def get_shell_cmd(self, *args):
        """
        On windows, changing directories can be complicated by the presence of 
        multiple drives. get_shell_cmd deals with this issue.
        """
        if sys.platform == 'win32': 
            command = ['%s'% os.path.splitdrive(self.texcache)[0]]
        else:
            command = []
        command.extend(args)
        return ' && '.join(command)
        
    def make_tex(self, tex, fontsize):
        basefile = self.get_basefile(tex, fontsize)
        texfile = '%s.tex'%basefile
        fh = file(texfile, 'w')
        fontcmd = {'sans-serif' : r'{\sffamily %s}',
                   'monospace'  : r'{\ttfamily %s}'}.get(self.font_family, 
                                                         r'{\rmfamily %s}')
        tex = fontcmd % tex
        s = r"""\documentclass{article}
%s
\usepackage[papersize={72in,72in}, body={70in,70in}, margin={1in,1in}]{geometry}
\pagestyle{empty}
\begin{document}
\fontsize{%f}{%f}%s
\end{document}
""" % (self._font_preamble, fontsize, fontsize*1.25, tex)
        fh.write(s)
        fh.close()
        
        return texfile
        
    def make_dvi(self, tex, fontsize, force=0):
        if debug: force = True

        basefile = self.get_basefile(tex, fontsize)
        dvifile = '%s.dvi'% basefile

        if force or not os.path.exists(dvifile):
            texfile = self.make_tex(tex, fontsize)
            outfile = basefile+'.output'
            command = self.get_shell_cmd('cd "%s"'% self.texcache, 
                            'latex -interaction=nonstopmode %s > "%s"'\
                            %(os.path.split(texfile)[-1], outfile))
            verbose.report(command, 'debug')
            exit_status = os.system(command)
            fh = file(outfile)
            if exit_status:
                raise RuntimeError('LaTeX was not able to process the flowing \
string:\n%s\nHere is the full report generated by LaTeX: \n\n'% tex + fh.read())
            else: verbose.report(fh.read(), 'debug')
            fh.close()
            for fname in glob.glob(basefile+'*'):
                if fname.endswith('dvi'): pass
                elif fname.endswith('tex'): pass
                else: os.remove(fname)

        return dvifile

    def make_png(self, tex, fontsize, dpi, force=0):
        if debug: force = True

        basefile = self.get_basefile(tex, fontsize, dpi)
        pngfile = '%s.png'% basefile

        # see get_rgba for a discussion of the background
        if force or not os.path.exists(pngfile):
            dvifile = self.make_dvi(tex, fontsize)
            outfile = basefile+'.output'
            command = self.get_shell_cmd('cd "%s"' % self.texcache, 
                        'dvipng -bg Transparent -D %s -T tight -o \
                        "%s" "%s" > "%s"'%(dpi, os.path.split(pngfile)[-1], 
                        os.path.split(dvifile)[-1], outfile))
            verbose.report(command, 'debug')
            exit_status = os.system(command)
            fh = file(outfile)
            if exit_status:
                raise RuntimeError('dvipng was not able to \
process the flowing file:\n%s\nHere is the full report generated by dvipng: \
\n\n'% dvifile + fh.read())
            else: verbose.report(fh.read(), 'debug')
            fh.close()
            os.remove(outfile)

        return pngfile

    def make_ps(self, tex, fontsize, force=0):
        if debug: force = True

        basefile = self.get_basefile(tex, fontsize)
        psfile = '%s.epsf'% basefile

        if force or not os.path.exists(psfile):
            dvifile = self.make_dvi(tex, fontsize)
            outfile = basefile+'.output'
            command = self.get_shell_cmd('cd "%s"'% self.texcache, 
                        'dvips -q -E -o "%s" "%s" > "%s"'\
                        %(os.path.split(psfile)[-1], 
                          os.path.split(dvifile)[-1], outfile))
            verbose.report(command, 'debug')
            exit_status = os.system(command)
            fh = file(outfile)
            if exit_status:
                raise RuntimeError('dvipng was not able to \
process the flowing file:\n%s\nHere is the full report generated by dvipng: \
\n\n'% dvifile + fh.read())
            else: verbose.report(fh.read(), 'debug')
            fh.close()
            os.remove(outfile)

        return psfile

    def get_ps_bbox(self, tex, fontsize):
        psfile = self.make_ps(tex, fontsize)
        ps = file(psfile)
        for line in ps:
            if line.startswith('%%BoundingBox:'):
                return [int(val) for val in line.split()[1:]]
        raise RuntimeError('Could not parse %s'%psfile)
        
    def get_rgba(self, tex, fontsize=None, dpi=None, rgb=(0,0,0)):
        """
        Return tex string as an rgba array
        """
        # dvipng assumes a constant background, whereas we want to
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
        if not fontsize: fontsize = rcParams['font.size']
        if not dpi: dpi = rcParams['savefig.dpi']
        r,g,b = rgb
        key = tex, fontsize, dpi, tuple(rgb)
        Z = self.arrayd.get(key)

        if Z is None:
            # force=True to skip cacheing while debugging
            pngfile = self.make_png(tex, fontsize, dpi, force=False)
            X = readpng(os.path.join(self.texcache, pngfile))

            if (self.dvipngVersion < '1.6') or rcParams['text.dvipnghack']:
                # hack the alpha channel as described in comment above
                alpha = sqrt(1-X[:,:,0])
            else:
                alpha = X[:,:,-1]

            Z = zeros(X.shape, Float)
            Z[:,:,0] = r
            Z[:,:,1] = g
            Z[:,:,2] = b
            Z[:,:,3] = alpha
            self.arrayd[key] = Z

        return Z
