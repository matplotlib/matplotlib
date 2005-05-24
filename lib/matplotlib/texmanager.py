"""
A class to manage TeX to figure conversion.

Requires tex to be installed on your system

For raster output, eg for the agg backend, this module
requires dvipng to be installed.


"""
import os, sys, md5
from matplotlib import get_home, get_data_path
from matplotlib._image import readpng, fromarray
from matplotlib.numerix import ravel, where, array, \
     zeros, Float, absolute, nonzero, sqrt
     
debug = False
     
class TexManager:
    """
    Convert strings to dvi files using TeX, caching the results to a
    working dir
    """
    path = get_home()
    if path is None: path = get_data_path()
    texcache = os.path.join(path, '.tex.cache')

    
    def __init__(self):
        if not os.path.isdir(self.texcache):
            os.mkdir(self.texcache)
        self.imaged = {}
        self.postscriptd = {}
        self.pscnt = 0
        self.dvipngVersion = None
        
    def make_dvi(self, tex, force=0):
        if debug: force = True
        
        prefix = self.get_prefix(tex)
        fname = os.path.join(self.texcache, prefix+ '.tex')
        dvitmp = prefix + '.dvi'
        dvifile = os.path.join(self.texcache, dvitmp)

        logfile = prefix + '.log'
        fh = file(fname, 'w')
        s = r"""\nopagenumbers
\hsize=72in
\vsize=72in
%s
\bye
""" % tex
        fh.write(s)
        fh.close()
        command = 'tex %s'%fname

        if force or not os.path.exists(dvifile):
            #sin, sout = os.popen2(command)
            #sout.close()
            os.system(command)
            os.rename(dvitmp, dvifile)
            os.remove(logfile)
        return dvifile


    def get_prefix(self, tex):
        return md5.md5(tex).hexdigest()
        
    def make_png(self, tex, dpi, force=0):
        if debug: force = True
        
        dvifile = self.make_dvi(tex)
        prefix = self.get_prefix(tex)
        pngfile = os.path.join(self.texcache, '%s_%d.png'% (prefix, dpi))

        
        command = "dvipng -bg Transparent -fg 'rgb 0.0 0.0 0.0' -D %d -T tight -o %s %s"% (dpi, pngfile, dvifile)

        #assume white bg
        #command = "dvipng -bg 'rgb 1.0 1.0 1.0' -fg 'rgb 0.0 0.0 0.0' -D %d -T tight -o %s %s"% (dpi, pngfile, dvifile)
        # assume gray bg
        #command = "dvipng -bg 'rgb 0.75 0.75 .75' -fg 'rgb 0.0 0.0 0.0' -D %d -T tight -o %s %s"% (dpi, pngfile, dvifile)                

        # see get_image for a discussion of the background
        if force or not os.path.exists(pngfile):
            os.system(command)
        return pngfile

    def make_ps(self, tex, dpi, force=0):
        if debug: force = True
        
        dvifile = self.make_dvi(tex)
        prefix = self.get_prefix(tex)
        psfile = os.path.join(self.texcache, '%s_%d.epsf'% (prefix, dpi))

        if not os.path.exists(psfile):
            command = "dvips -E -D %d -o %s %s"% (dpi, psfile, dvifile)
            os.system(command)

        return psfile

    def get_ps_bbox(self, tex):
        key = tex
        val = self.postscriptd.get(key)
        if val is not None: return val
        psfile = self.make_ps(tex, dpi=72.27)
        ps = file(psfile).read()
        for line in ps.split('\n'):
            if line.startswith('%%BoundingBox:'):
                return [int(val) for val in line.split()[1:]]
        raise RuntimeError('Could not parse %s'%psfile)
        
        
    def __get_ps(self, tex, fontsize=10, dpi=80, rgb=(0,0,0)):
        """
        Return bbox, header, texps for tex string via make_ps    
        """

        # this is badly broken and safe to ignore.
        key = tex, fontsize, dpi, rgb
        val = self.postscriptd.get(key)
        if val is not None: return val
        psfile = self.make_ps(tex, dpi)
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
        
    def get_image(self, tex, fontsize=10, dpi=80, rgb=(0,0,0)):
        """
        Return tex string as a matplotlib._image.Image
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
        
        # assuming standard 10pt design size space
        dpi = fontsize/10.0 * dpi
        
        r,g,b = rgb
        key = tex, dpi, tuple(rgb)
        im = self.imaged.get(key)


        if im is None:
            # force=True to skip cacheing while debugging
            pngfile = self.make_png(tex, dpi, force=False) 
            X = readpng(pngfile)

            vers = self.get_dvipng_version()
            if vers<'1.6':
                alpha = sqrt(1-X[:,:,0])
            else:
                # 1.6 has the alpha channel right
                alpha = sqrt(X[:,:,-1])
            
            #from matplotlib.mlab import prctile
            #print 'ptile', prctile(ravel(X[:,:,0])), prctile(ravel(X[:,:,-1]))

            Z = zeros(X.shape, Float)
            Z[:,:,0] = r
            Z[:,:,1] = g
            Z[:,:,2] = b
            Z[:,:,3] = alpha
            im = fromarray(Z, 1)
               
            self.imaged[key] = im
        return im

    def get_dvipng_version(self):
        if self.dvipngVersion is not None: return self.dvipngVersion
        sin, sout = os.popen2('dvipng --version')
        for line in sout.readlines():
            if line.startswith('dvipng '):
                self.dvipngVersion = line.split()[-1]
                return self.dvipngVersion
        raise RuntimeError('Could not obtain dvipng version')
            
