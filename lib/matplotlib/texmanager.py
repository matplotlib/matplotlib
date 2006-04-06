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
from subprocess import Popen, STDOUT, PIPE
from matplotlib import get_configdir, get_home, get_data_path, \
     rcParams, verbose
from matplotlib._image import readpng
from matplotlib.numerix import ravel, where, array, \
     zeros, Float, absolute, nonzero, sqrt
     
debug = False
     
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

    os.environ['TEXMFOUTPUT'] = texcache
    if os.path.exists(oldcache):
        print >> sys.stderr, """\
WARNING: found a TeX cache dir in the deprecated location "%s".
  Moving it to the new default location "%s"."""%(oldcache, texcache)
        shutil.move(oldcache, texcache)

    dvipngVersion = None

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
        self.dvipngVersion = self.get_dvipng_version()
        
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
        
    def get_prefix(self, tex, fontsize, dpi=None):
        s = tex + self._fontconfig + ('%f'%fontsize)
        if dpi: s += ('%s'%dpi)
        return md5.md5(s).hexdigest()
        
    def get_font_config(self):
        return self._fontconfig
        
    def get_font_preamble(self):
        return self._font_preamble
        
    def make_tex(self, tex, prefix, fontsize):
        texfile = os.path.join(self.texcache,prefix+'.tex')
        fh = file(texfile, 'w')
        fontcmd = {'sans-serif' : r'{\sffamily %s}',
            'monospace'  : r'{\ttfamily %s}'}.get(self.font_family, 
                r'{\rmfamily %s}')
        tex = fontcmd % tex
        s = r"""\documentclass[10pt]{article}
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
        
        prefix = self.get_prefix(tex, fontsize)
        dvibase = prefix + '.dvi'
        dvifile = os.path.join(self.texcache, dvibase)

        if force or not os.path.exists(dvifile):
            texfile = self.make_tex(tex, prefix, fontsize)
            command = 'latex -interaction=nonstopmode "%s"'%texfile
            verbose.report(command, 'debug-annoying')
            process = Popen([command], shell=True, stdin=PIPE, stdout=PIPE,\
                            stderr=STDOUT)
            exit_status = process.wait()
            if exit_status: raise RuntimeError('LaTeX was not able to process \
the flowing string:\n%s\nHere is the full report generated by LaTeX: \
\n\n'% tex + process.stdout.read())
            else: verbose.report(process.stdout.read(), 'debug-annoying')
##            verbose.report(command, 'debug-annoying')
##            stdin, stdout, stderr = os.popen3(command)
##            verbose.report(stdout.read(), 'debug-annoying')
##            err = stderr.read()
##            if err: verbose.report(err, 'helpful')

        # tex will put it's output in the current dir if possible, and
        # if not in TEXMFOUTPUT.  So check for existence in current
        # dir and move it if necessary and then cleanup
        if os.path.exists(dvibase):
            shutil.move(dvibase, dvifile)
            for fname in glob.glob(prefix+'*'):
                os.remove(fname)
        return dvifile
        
    def make_png(self, tex, fontsize, dpi, force=0):
        if debug: force = True

        dvifile = self.make_dvi(tex, fontsize)
        prefix = self.get_prefix(tex, fontsize, dpi)
        pngfile = os.path.join(self.texcache, '%s.png'% prefix)
        
        command = 'dvipng -bg Transparent -D "%s" -T tight -o "%s" "%s"'%\
                  (dpi, pngfile, dvifile)

        # see get_rgba for a discussion of the background
        if force or not os.path.exists(pngfile):
            verbose.report(command, 'debug-annoying')
            process = Popen([command], shell=True, stdin=PIPE, stdout=PIPE, 
                            stderr=STDOUT)
            exit_status = process.wait()
            if exit_status: raise RuntimeError('dvipng was not able to \
process the flowing file:\n%s\nHere is the full report generated by dvipng: \
\n\n'% dvifile + process.stdout.read())
            else: verbose.report(process.stdout.read(), 'debug-annoying')
##            stdin, stdout, stderr = os.popen3(command)
##            verbose.report(stdout.read(), 'debug-annoying')
##            err = stderr.read()
##            if err: verbose.report(err, 'helpful')
        return pngfile

    def make_ps(self, tex, fontsize, force=0):
        if debug: force = True
        
        dvifile = self.make_dvi(tex, fontsize)
        prefix = self.get_prefix(tex, fontsize)
        psfile = os.path.join(self.texcache, '%s.epsf'% prefix)

        if not os.path.exists(psfile):
            command = 'dvips -q -E -o "%s" "%s"'% (psfile, dvifile)
            process = Popen([command], shell=True, stdin=PIPE, stdout=PIPE,
                            stderr=STDOUT)
            exit_status = process.wait()
            if exit_status: raise RuntimeError('dvips was not able to process \
the flowing file:\n%s\nHere is the full report generated by dvips: \
\n\n'% dvifile + process.stdout.read())
            else: verbose.report(process.stdout.read(), 'debug-annoying')
##            verbose.report(command, 'debug-annoying')
##            stdin, stdout, stderr = os.popen3(command)
##            verbose.report(stdout.read(), 'debug-annoying')
##            err = stderr.read()
##            if err: verbose.report(err, 'helpful')

        return psfile

    def get_ps_bbox(self, tex, fontsize):
        key = tex
        val = self.postscriptd.get(key)
        if val is not None: return val
        psfile = self.make_ps(tex, fontsize)
        ps = file(psfile).read()
        for line in ps.split('\n'):
            if line.startswith('%%BoundingBox:'):
                return [int(val) for val in line.split()[1:]]
        raise RuntimeError('Could not parse %s'%psfile)
        
        
    def __get_ps(self, tex, fontsize=10, rgb=(0,0,0)):
        """
        Return bbox, header, texps for tex string via make_ps    
        """

        # this is badly broken and safe to ignore.
        key = tex, fontsize, dpi, rgb
        val = self.postscriptd.get(key)
        if val is not None: return val
        psfile = self.make_ps(tex, fontsize)
        ps = file(psfile).read()
        
        # parse the ps
        bbox = None
        header = []
        tex = []
        inheader = False
        texon = False
        fonts = []
        infont = False
        replaced = {}
        for line in ps.split('\n'):
            if line.startswith('%%EndProlog'):
                inheader = False
            if line.startswith('%%Trailer'):
                break

            if line.startswith('%%BoundingBox:'):
                bbox = [int(val) for val in line.split()[1:]]
                continue
            if line.startswith('%%BeginFont:'):
                fontname = line.split()[-1].strip()
                newfontname = fontname + str(self.pscnt)
                replaced[fontname] = newfontname
                thisfont = [line]
                infont = True
                continue
            if line.startswith('%%EndFont'):
                thisfont.append('%%EndFont\n')
                fonts.append('\n'.join(thisfont).replace(fontname, newfontname))
                thisfont = []
                infont = False                
                continue
            if infont:
                thisfont.append(line)
                continue
            if line.startswith('%%BeginProcSet:'):                
                inheader = True
            if inheader:
                header.append(line)
            if line.startswith('%%EndSetup'):
                assert(not inheader)
                texon = True
                continue



            if texon:
                line = line.replace('eop end', 'end')
                tex.append(line)

        def clean(s):
            for k,v in replaced.items():
                s = s.replace(k,v)
            return s
        
        header.append('\n')
        tex.append('\n')
        if bbox is None:
            raise RuntimeError('Failed to parse dvips file: %s' % psfile)
        
        replaced['TeXDict'] = 'TeXDict%d'%self.pscnt
        header = clean('\n'.join(header))
        tex = clean('\n'.join(tex))
        fonts = '\n'.join(fonts)
        val = bbox, header, fonts, tex
        self.postscriptd[key] = val

        self.pscnt += 1
        return val
        
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
            X = readpng(pngfile)
            vers = self.get_dvipng_version()
            #print 'dvipng version', vers
            if vers<'1.6' or rcParams['text.dvipnghack']:
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

    def get_dvipng_version(self):
        if self.dvipngVersion is not None: return self.dvipngVersion
        process = Popen(['dvipng --version'], shell=True, stdin=PIPE,
                        stderr=PIPE, stdout=PIPE)
        exit_status = process.wait()
        if exit_status: raise RuntimeError('Could not obtain dvipng version\n\
\n\n' + process.stdout.read())
        else: sout = process.stdout.read()
        for line in sout.split('\n'):
##        sin, sout = os.popen2('dvipng --version')
##        for line in sout.readlines():
            if line.startswith('dvipng '):
                version = line.split()[-1]
                verbose.report('Found dvipng version %s'% version, 
                    'helpful')
                return version
        raise RuntimeError('Could not obtain dvipng version')
            
