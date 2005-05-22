import os, sys, time
from cStringIO import StringIO
from matplotlib import verbose, __version__, rcParams, get_data_path
from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureManagerBase, FigureCanvasBase
from backend_ps import RendererPS, FigureCanvasPS, psDefs, defaultPaperType, \
     defaultPaperSize, _nums_to_str
from matplotlib.texmanager import TexManager

class RendererLatex(RendererPS):

    def __init__(self, *args, **kwargs):
        RendererPS.__init__(self, *args, **kwargs)
        self.textcnt = 0
        self.psfrag = []
        self.texmanager = TexManager()
        

    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop
        """
        l,b,r,t = self.texmanager.get_ps_bbox(s)
        w = r-l
        h = t-b
        #print s, w, h
        return w, h

    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        """
        draw a Text instance
        """
        w, h = self.get_text_width_height(s, prop, ismath)
        fontsize = prop.get_size_in_points()
        corr = w/2*(fontsize-10)/10
        pos = _nums_to_str(x-corr, y)
        thetext = 'psmarker%d' % self.textcnt
        setcolor = '%1.3f %1.3f %1.3f setrgbcolor' % gc.get_rgb()
        scale = float(fontsize/10.0)
        color = r'\rgb %1.3f %1.3f %1.3f'%gc.get_rgb()
        tex = '\color{rgb}{%s}'%s
        self.psfrag.append(r'\psfrag{%s}[bl][bl][%f][%f]{%s}'%(thetext, scale, angle, s))
        ps = """\
gsave
%(pos)s moveto
%(setcolor)s
(%(thetext)s)
show
grestore
    """ % locals()

        self._pswriter.write(ps)
        self.textcnt += 1




class FigureCanvasLatex(FigureCanvasBase):
    basepath = get_data_path()

    def draw(self):
        pass
    
    def print_figure(self, outfile, dpi=72,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):
        """
        Render the figure to hardcopy.  Set the figure patch face and
        edge colors.  This is useful because some of the GUIs have a
        gray figure face color background and you'll probably want to
        override this on hardcopy

        The output file is a psfrag latex file
        """

        basename, ext = os.path.splitext(outfile)
        if not ext.endswith('tex'): outfile += '.tex'
        psname = basename + '.eps'
        psh = file(psname, 'w')
        latexh = file(outfile, 'w')

        # center the figure on the paper
        self.figure.dpi.set(72)        # ignore the passsed dpi setting for PS
        width, height = self.figure.get_size_inches()

        if orientation=='landscape':
            isLandscape = True
            paperHeight, paperWidth = defaultPaperSize
        else:
            isLandscape = False
            paperWidth, paperHeight = defaultPaperSize

        xo = 72*0.5*(paperWidth - width)
        yo = 72*0.5*(paperHeight - height)
        l, b, w, h = self.figure.bbox.get_bounds()

        llx = xo
        lly = yo
        urx = llx + w
        ury = lly + h

        if isLandscape:
            xo, yo = 72*paperHeight - yo, xo
            llx, lly, urx, ury = lly, llx, ury, urx
            rotation = 90
        else:
            rotation = 0

        # generate PostScript code for the figure and store it in a string
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        self._pswriter = StringIO()
        renderer = RendererLatex(width, height, self._pswriter)
        self.figure.draw(renderer)

        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)

        # write the PostScript headers

        print >>psh, "%!PS-Adobe-3.0 EPSF-3.0"
        print >>psh, "%%Title: "+psname
        print >>psh, ("%%Creator: matplotlib version "
                     +__version__+", http://matplotlib.sourceforge.net/")
        print >>psh, "%%CreationDate: "+time.ctime(time.time())
        print >>psh, "%%%%BoundingBox: %d %d %d %d" % (llx, lly, urx, ury)
        print >>psh, "%%EndComments"

        Ndict = len(psDefs)
        print >>psh, "/mpldict %d dict def"%Ndict
        print >>psh, "mpldict begin"

        for d in psDefs:
            d=d.strip()
            for l in d.split('\n'):
                print >>psh, l.strip()

        print >>psh, "mpldict begin"
        #print >>psh, "gsave"
        print >>psh, "%s translate"%_nums_to_str(xo, yo)
        if rotation:
            print >>psh, "%d rotate"%rotation
        print >>psh, "%s clipbox"%_nums_to_str(width*72, height*72, 0, 0)

        # write the figure
        print >>psh, self._pswriter.getvalue()

        # write the trailer
        #print >>psh, "grestore"
        print >>psh, "end"
        print >>psh, "showpage"

        psh.close()


        print >>latexh, r"""\documentclass{article}
\usepackage{psfrag}
\usepackage[dvips]{graphicx}
\pagestyle{empty}
\begin{document}


\begin{figure}[t]
  %s
  \resizebox{5.5in}{!}{\includegraphics{%s}}
 
\end{figure}

\end{document}
"""% ('\n'.join(renderer.psfrag), psname)


def new_figure_manager(num, *args, **kwargs):
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasLatex(thisFig)
    manager = FigureManagerBase(canvas, num)
    return manager


