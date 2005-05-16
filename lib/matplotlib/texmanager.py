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
     zeros, Float, absolute, nonzero, put, putmask
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
        self.images = {}

    def make_dvi(self, tex):
        prefix = self.get_prefix(tex)
        fname = os.path.join(self.texcache, prefix+ '.tex')
        dvitmp = prefix + '.dvi'
        dvifile = os.path.join(self.texcache, dvitmp)

        logfile = prefix + '.log'
        fh = file(fname, 'w')
        fh.write(tex + '\n\\nopagenumbers\n\\bye\n')
        fh.close()
        command = 'tex %s'%fname
        if not os.path.exists(dvifile):
            #sin, sout = os.popen2(command)
            #sout.close()
            os.system(command)
            os.rename(dvitmp, dvifile)
            os.remove(logfile)
        return dvifile

    def get_prefix(self, tex):
        return md5.md5(tex).hexdigest()
        
    def make_png(self, tex, dpi, force=0):
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
        im = self.images.get(key)


        if im is None:
            # skip cacheing while debugging
            pngfile = self.make_png(tex, dpi, force=True) 
            X = readpng(pngfile)
            # To compare my results with the result of using dvipng's,
            # change this to 'if 0:' and uncomment the third "command"
            # variable in make_png.  The black text for the tick
            # labels on the gray background look better with the
            # dvipng rendering, so I must be making some error in how
            # I try and recover the alpha information
            if 1:                
                alpha = 1-X[:,:,0]
                visible = alpha>0
                print 'min/max', min(ravel(alpha)), max(ravel(alpha))
                Z = zeros(X.shape, Float)
                Z[:,:,0] = r
                Z[:,:,1] = g
                Z[:,:,2] = b
                Z[:,:,3] = alpha
                im = fromarray(Z, 1)
            else:
                alpha = X[:,:,-1]
                from matplotlib.mlab import prctile
                print 'ptile', prctile(ravel(alpha))
                im = fromarray(X, 1)
               
            self.images[key] = im
        return im
       
