# -*- coding: iso-8859-1 -*-
"""
A PDF matplotlib backend
Author: Jouni K Seppänen <jks@iki.fi>

As of yet, this implements a small subset of the backend protocol, but
enough to get some output from simple plots. Alpha is supported.
"""
from __future__ import division

import md5
import re
import sys
import time
import zlib

from math import ceil, cos, floor, pi, sin

from matplotlib import __version__, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import enumerate, is_string_like
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.ft2font import FT2Font, FIXED_WIDTH, ITALIC, LOAD_NO_SCALE
from matplotlib.transforms import Bbox


def tmap(*args):
    """Call map() and convert result to a tuple."""
    return tuple(map(*args))

def fill(strings, line=75):
    """Make one string from sequence of strings, breaking lines 
    to form lines of at most 72 characters, if possible."""

    s, strings = [strings[0]], strings[1:]
    while strings:
	if len(s[-1]) + len(strings[0]) < 72:
	    s[-1] += ' ' + strings[0]
	else:
	    s.append(strings[0])
	strings = strings[1:]
    return '\n'.join(s)
    

def pdfRepr(obj):
    """Map Python objects to PDF syntax."""

    # Some objects defined later have their own pdfRepr method.
    if 'pdfRepr' in dir(obj):
	return obj.pdfRepr()

    # Floats. PDF does not have exponential notation (1.0e-10) so we
    # need to use %f with some precision.  Perhaps the precision
    # should adapt to the magnitude of the number?
    elif isinstance(obj, float):
	r = "%.10f" % obj
	return r.rstrip('0').rstrip('.')

    # Integers are written as such.
    elif isinstance(obj, (int, long)):
	return "%d" % obj

    # Strings are written in parentheses, with backslashes and parens
    # escaped. Actually balanced parens are allowed, but it is
    # simpler to escape them all. TODO: cut long strings into lines;
    # I believe there is some maximum line length in PDF.
    elif is_string_like(obj):
	return '(' + re.sub(r'([\\()])', r'\\\1', obj) + ')'

    # Dictionaries. The keys must be PDF names, so if we find strings
    # there, we make Name objects from them. The values may be
    # anything, so the caller must ensure that PDF names are
    # represented as Name objects.
    elif isinstance(obj, dict):
	r = ["<<"]
	r.extend(["%s %s" % (Name(key).pdfRepr(), 
			     pdfRepr(val))
		  for key, val in obj.items()])
	r.append(">>")
	return fill(r)

    # Lists.
    elif isinstance(obj, (list, tuple)):
	r = ["["]
	r.extend([pdfRepr(val) for val in obj])
	r.append("]")
	return fill(r)

    # Booleans.
    elif isinstance(obj, bool):
	return ['false', 'true'][obj]

    # The null keyword.
    elif obj is None:
	return 'null'

    else:
	raise TypeError, \
	    "Don't know a PDF representation for %s objects." \
	    % type(obj)

class Reference:
    """PDF reference object. 
    Use PdfFile.reserveObject() to create References.
    """

    def __init__(self, id):
	self.id = id

    def pdfRepr(self):
	return "%d 0 R" % self.id

    def write(self, contents, file):
	write = file.write
	write("%d 0 obj\n" % self.id)
	write(pdfRepr(contents))
	write("\nendobj\n")

class Name:
    """PDF name object."""

    def __init__(self, name):
	if isinstance(name, Name):
	    self.name = name.name
	else:
	    def hexify(match):
		return '#%02x' % ord(match.group())
	    self.name = re.sub(r'[^!-~]', hexify, name)

    def pdfRepr(self):
	return '/' + self.name

class Stream:
    """PDF stream object.

    This has no pdfRepr method. Instead, call begin(), then output the
    contents of the stream by calling write(), and finally call end().
    """

    def __init__(self, id, len, file):
	"""id: object id of stream; len: an unused Reference object
	for the length of the stream; file: a PdfFile
	"""
	self.id = id		# object id
	self.len = len		# id of length object
	self.file = file	# file to which the stream is written
	self.compressobj = None	# compression object

    def begin(self):
	"""Initialize stream."""

	write = self.file.fh.write
	self.file.recordXref(self.id)
	write("%d 0 obj\n" % self.id)
	dict = { 'Length': self.len }
	if rcParams['pdf.compression']:
	    dict['Filter'] = Name('FlateDecode')
	write(pdfRepr(dict))
	write("\nstream\n")
	self.pos = self.file.fh.tell()
	if rcParams['pdf.compression']:
	    self.compressobj = zlib.compressobj(rcParams['pdf.compression'])

    def end(self):
	"""Finalize stream."""

	self._flush()
	length = self.file.fh.tell() - self.pos
	self.file.write("\nendstream\nendobj\n")
	self.file.writeObject(self.len, length)

    def write(self, data):
	"""Write some data on the stream."""

	if self.compressobj is None:
	    self.file.fh.write(data)
	else:
	    compressed = self.compressobj.compress(data)
	    self.file.fh.write(compressed)

    def _flush(self):
	"""Flush the compression object."""

	if self.compressobj is not None:
	    compressed = self.compressobj.flush()
	    self.file.fh.write(compressed)
	    self.compressobj = None

class PdfFile:
    """PDF file with one page."""

    def __init__(self, width, height, filename):
	self.nextObject = 1	# next free object id
	self.xrefTable = [ [0, 65535, 'the zero object'] ]
	fh = file(filename, 'wb')
	self.fh = fh
	self.currentstream = None # stream object to write to, if any
	fh.write("%PDF-1.4\n")	  # 1.4 is the first version to have alpha
	# Output some binary chars as a comment so various utilities
	# recognize the file as binary by looking at the first few
	# lines (see note in section 3.4.1 of the PDF reference).
	fh.write("%\254\334 \253\272\n")

	self.rootObject = self.reserveObject('root')
	self.infoObject = self.reserveObject('info')
	pagesObject = self.reserveObject('pages')
	thePageObject = self.reserveObject('page 0')
	contentObject = self.reserveObject('contents of page 0')
	self.fontObject = self.reserveObject('fonts')
	self.alphaStateObject = self.reserveObject('alpha states')
	resourceObject = self.reserveObject('resources')

	root = { 'Type': Name('Catalog'), 
		 'Pages': pagesObject }
	self.writeObject(self.rootObject, root)

	info = { 'Producer': 'matplotlib version ' + __version__ \
		 + ', http://matplotlib.sourceforge.net', }
	# Possible TODO: Title, Author, Subject, Keywords, CreationDate
	self.writeObject(self.infoObject, info)

	pages = { 'Type': Name('Pages'), 
		  'Kids': [ thePageObject ],
		  'Count': 1 }
	self.writeObject(pagesObject, pages)

	thePage = { 'Type': Name('Page'),
		    'Parent': pagesObject,
		    'Resources': resourceObject,
		    'MediaBox': [ 0, 0, 72*width, 72*height ],
		    'Contents': contentObject }
	self.writeObject(thePageObject, thePage)

	# self.fontNames maps filenames to internal font names
	self.fontNames = {}
	self.nextFont = 1	# next free internal font name

	self.alphaStates = {}	# maps alpha values to graphics state objects
	self.nextAlphaState = 1

	# The PDF spec recommends to include every procset
	procsets = [ Name(x)
		     for x in "PDF Text ImageB ImageC ImageI".split() ]

	# Write resource dictionary.
	# Possibly TODO: more general ExtGState (graphics state dictionaries)
	#                ColorSpace Pattern Shading XObject Properties
	resources = { 'Font': self.fontObject,
		      'ExtGState': self.alphaStateObject,
		      'ProcSet': procsets }
	self.writeObject(resourceObject, resources)

	# Start the content stream of the page
	self.contents = \
	    Stream(contentObject.id, 
		   self.reserveObject('length of content stream'),
		   self)
	self.contents.begin()
	self.currentstream = self.contents

    def close(self):
	# End the content stream and write out the various deferred
	# objects
	self.contents.end()
	self.writeFonts()
	self.writeObject(self.alphaStateObject, 
			 dict([(val[0], val[1]) 
			       for val in self.alphaStates.values()]))
	self.writeXref()
	self.writeTrailer()
	self.fh.close()

    def write(self, data):
	if self.currentstream is None:
	    self.fh.write(data)
	else:
	    self.currentstream.write(data)

    # These fonts do not need to be embedded; every PDF viewing
    # application is required to have them.
    base14 = [ 'Times-Roman', 'Times-Bold', 'Times-Italic',
	       'Times-BoldItalic', 'Symbol', 'ZapfDingbats' ] + \
	     [ prefix + postfix 
	       for prefix in 'Helvetica', 'Courier'
	       for postfix in '', '-Bold', '-Oblique', '-BoldOblique' ]

    def fontName(self, fontprop):
	filename = fontManager.findfont(fontprop)
	Fx = self.fontNames.get(filename, None)
	if Fx is None:
	    Fx = Name('F%d' % self.nextFont)
	    self.fontNames[filename] = Fx
	    self.nextFont += 1
	return Fx

    def writeFonts(self):
	fonts = {}
	for filename, Fx in self.fontNames.items():
	    # TODO: The following test is wrong, since findfont really
	    # returns a file name. Think about how to allow users to
	    # specify the base14 fonts (does fontManager know anything
	    # about them?).
	    if filename in self.base14:
		fontdict = { 'Subtype': Name('Type1'),
			     'BaseFont': Name(filename) }
		# etc...
	    else:
		fontdictObject = self.embedTTF(filename, )
	    fonts[Fx] = fontdictObject
	    #print >>sys.stderr, filename

	self.writeObject(self.fontObject, fonts)

    def embedTTF(self, filename):
	"""Embed the TTF font from the named file into the document."""

	font = FT2Font(filename)

	def convert(length, upe=font.units_per_EM, nearest=True):
	    "Convert font coordinates to PDF glyph coordinates"
	    value = length / upe * 1000
	    if nearest: return round(value)
	    # Perhaps best to round away from zero for bounding
	    # boxes and the like
	    if value < 0: return floor(value)
	    else: return ceil(value)

	# You are lost in a maze of TrueType tables, all different...
	ps_name = Name(font.get_sfnt()[(1,0,0,6)])
	pclt = font.get_sfnt_table('pclt') \
	    or { 'capHeight': 0, 'xHeight': 0 }
	post = font.get_sfnt_table('post') \
	    or { 'italicAngle': (0,0) }
	ff = font.face_flags
	sf = font.style_flags
	charmap = font.get_charmap()
	chars = sorted(charmap.keys())
	# TODO: the widths are wrong (Adobe Reader complains)
	widths = [ convert(font.load_char(i, flags=LOAD_NO_SCALE)
			   .horiAdvance)
		   for i in range(chars[0], chars[-1]+1) ]

 	fontdict = { 'Type': Name('Font'),
		     'Subtype': Name('TrueType'),
		     'BaseFont': ps_name,
		     'FirstChar': chars[0],
		     'LastChar': chars[-1],
		     'Widths': self.reserveObject('font widths'),
		     'FontDescriptor': 
		       self.reserveObject('font descriptor') }
	# TODO: Encoding?

	flags = 0
	if ff & FIXED_WIDTH: flags |= 1 << 0
	if 0: flags |= 1 << 1 # TODO: serif
	if 0: flags |= 1 << 2 # TODO: symbolic
	else: flags |= 1 << 5 # TODO: nonsymbolic
	if sf & ITALIC: flags |= 1 << 6
	if 0: flags |= 1 << 16 # TODO: all caps
	if 0: flags |= 1 << 17 # TODO: small caps
	if 0: flags |= 1 << 18 # TODO: force bold

	descriptor = { 
	    'Type': Name('FontDescriptor'),
	    'FontName': ps_name,
	    'Flags': flags,
	    'FontBBox': [ convert(x, nearest=False) for x in font.bbox ],
	    'Ascent': convert(font.ascender, nearest=False),
	    'Descent': convert(font.descender, nearest=False),
	    'CapHeight': convert(pclt['capHeight'], nearest=False),
	    'XHeight': convert(pclt['xHeight']),
	    'ItalicAngle': post['italicAngle'][1], # ???
	    'FontFile2': self.reserveObject('font file'),
	    'StemV': 0 # ???
	    }

	# Other FontDescriptor keys include:
	# /FontFamily /Times (optional)
	# /FontStretch /Normal (optional)
	# /FontFile (stream for type 1 font)
	# /CharSet (used when subsetting type1 fonts)

	fontdictObject = self.reserveObject('font dictionary')
	self.writeObject(fontdictObject, fontdict)
	self.writeObject(fontdict['Widths'], widths)
	self.writeObject(fontdict['FontDescriptor'], descriptor)
	self.currentstream = \
	    Stream(descriptor['FontFile2'].id,
		   self.reserveObject('length of font stream'),
		   self)
	self.currentstream.begin()
	fontfile = open(filename, 'rb')
	while True:
	    data = fontfile.read(4096)
	    if not data: break
	    self.currentstream.write(data)
	self.currentstream.end()
	self.currentstream = None

	return fontdictObject

    def alphaState(self, alpha):
	"""Return name of an ExtGState that sets alpha to the given value"""
	
	state = self.alphaStates.get(alpha, None)
	if state is not None:
	    return state[0]
	
	name = Name('A%d' % self.nextAlphaState)
	self.nextAlphaState += 1
	self.alphaStates[alpha] = \
	    (name, { 'Type': Name('ExtGState'),
		     'CA': alpha, 'ca': alpha })
	return name

    def reserveObject(self, name=''):
	"""Reserve an ID for an indirect object.
	The name is used for debugging in case we forget to print out
	the object with writeObject.
	"""

	id = self.nextObject
	self.nextObject += 1
	self.xrefTable.append([None, 0, name])
	return Reference(id)

    def recordXref(self, id):
	self.xrefTable[id][0] = self.fh.tell()

    def writeObject(self, object, contents):
	self.recordXref(object.id)
	object.write(contents, self)

    def writeXref(self):
	"""Write out the xref table."""

	self.startxref = self.fh.tell()
	self.write("xref\n0 %d\n" % self.nextObject)
	i = 0
	borken = False
	for offset, generation, name in self.xrefTable:
	    if offset is None:
		print >>sys.stderr, \
		    'No offset for object %d (%s)' % (i, name)
		borken = True
	    else:
		self.write("%010d %05d n \n" % (offset, generation))
	    i += 1
	if borken:
	    raise AssertionError, 'Indirect object does not exist'

    def writeTrailer(self):
	"""Write out the PDF trailer."""

	self.write("trailer\n")
	self.write(pdfRepr(
		{'Size': self.nextObject,
		 'Root': self.rootObject, 
		 'Info': self.infoObject }))
	# Could add 'Info' and 'ID'
	self.write("\nstartxref\n%d\n%%%%EOF\n" % self.startxref)



class RendererPdf(RendererBase):

    def __init__(self, file):
	self.file = file
	self.gc = self.new_gc()

    def check_gc(self, gc):
	delta = self.gc.delta(gc)
	if delta:
	    self.file.write('%s\n' % delta)
	    self.gc.copy_properties(gc)

    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, angle1, angle2):
        print >>sys.stderr, "draw_arc called"
    
    def draw_image(self, x, y, im, bbox):
        print >>sys.stderr, "draw_image called"
    
    def draw_line(self, gc, x1, y1, x2, y2):
	self.check_gc(gc)
        self.file.write('%s %s m %s %s l S\n' % 
			tmap(pdfRepr, (x1, y1, x2, y2)))
    
    def draw_lines(self, gc, x, y):
	write = self.file.write
	pr = pdfRepr

	self.check_gc(gc)
	write('%s %s m\n' % (pr(x[0]), pr(y[0])))
	for i in range(1,len(x)):
	    write('%s %s l\n' % (pr(x[i]), pr(y[i])))
	write('S\n')

    def draw_point(self, gc, x, y):
        print >>sys.stderr, "draw_point called"

    def draw_polygon(self, gcEdge, rgbFace, points):
	write = self.file.write
	pr = pdfRepr

	self.check_gc(gcEdge)
	write('%s %s m\n' % (pr(points[0][0]),
			     pr(points[0][1])))
	for x,y in points[1:]:
	    write('%s %s l\n' % (pr(x), pr(y)))
	write('q\n%s %s %s rg\nb\nQ\n' % tmap(pr, rgbFace))

    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
        print >>sys.stderr, "draw_rectangle called"
        pass

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
	# TODO: fix positioning
	#       combine consecutive texts into one BT/ET delimited section
	#       mathtext
	self.check_gc(gc)
	fontName = self.file.fontName(prop)
	self.file.write('BT\n%s %s Tf\n' % 
			(pdfRepr(fontName), 
			 pdfRepr(prop.get_size_in_points())))
	if angle == 0:
	    self.file.write('%s %s Td\n' % tmap(pdfRepr, (x,y)))
	else:
	    angle = angle / 180.0 * pi
	    self.file.write('%s %s %s %s %s %s Tm\n' % 
			    tmap(pdfRepr, 
				 ( cos(angle), sin(angle),
				  -sin(angle), cos(angle),
				   x,          y         )))
					    
	self.file.write('%s Tj\nET\n' % pdfRepr(s))
         
    def flipy(self):
        return False
    
    def get_canvas_width_height(self):
        return 100, 100

    def get_text_width_height(self, s, prop, ismath):
        return 1, 1
                              
    def new_gc(self):
        return GraphicsContextPdf(self.file)

    def points_to_pixels(self, points):
        return points


class GraphicsContextPdf(GraphicsContextBase):

    def __init__(self, file):
	GraphicsContextBase.__init__(self)
	self.file = file

    capstyles = { 'butt': 0, 'round': 1, 'projecting': 2 }
    joinstyles = { 'miter': 0, 'round': 1, 'bevel': 2 }

    def capstyle_cmd(self, style):
	return '%s J' % pdfRepr(self.capstyles[style])

    def joinstyle_cmd(self, style):
	return '%s j' % pdfRepr(self.joinstyles[style])

    def linewidth_cmd(self, width):
	return '%s w' % pdfRepr(width)

    def dash_cmd(self, dashes):
	offset, dash = GraphicsContextPdf.dashd[style]
	return '%s %s d' % (pdfRepr(list(dash)), offset)

    def alpha_cmd(self, alpha):
	name = self.file.alphaState(alpha)
	return '%s gs' % pdfRepr(name)

    def rgb_cmd(self, rgb):
	# setting both fill and stroke colors, is that right?
	rgb = tmap(pdfRepr, rgb)
	return ('%s %s %s RG ' % rgb) + ('%s %s %s rg' % rgb)

    commands = { 
	'_alpha': alpha_cmd,
	'_capstyle': capstyle_cmd,
	'_joinstyle': joinstyle_cmd,
	'_linewidth': linewidth_cmd,
	'_dashes': dash_cmd,
	'_rgb': rgb_cmd,
	}

    # TODO: _cliprect, _linestyle, _hatch
    # _cliprect needs pushing/popping the graphics state,
    # probably needs to be done in RendererPdf
    
    def delta(self, other):
	"""What PDF commands are needed to transform self into other?
	"""
	cmds = []
	for param in self.commands.keys():
	    if getattr(self, param) != getattr(other, param):
		cmd = self.commands[param]
		cmds.append(cmd(self, getattr(other, param)))
	return '\n'.join(cmds)
		      
        
########################################################################
#    
# The following functions and classes are for pylab and implement
# window/figure managers, etc...
#
########################################################################


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    # if a main-level app must be created, this is the usual place to
    # do it -- see backend_wx, backend_wxagg and backend_tkagg for
    # examples.  Not all GUIs require explicit instantiation of a
    # main-level app (egg backend_gtk, backend_gtkagg) for pylab

    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasPdf(thisFig)
    manager = FigureManagerPdf(canvas, num)
    return manager


class FigureCanvasPdf(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
    """

#     def __init__(self, figure):
# 	FigureCanvasBase.__init__(self, figure)

    def draw(self):
	pass
        
    def print_figure(self, filename, dpi=72, facecolor='w', edgecolor='w',
                     orientation='portrait', **kwargs):
        """
        Render the figure to hardcopy. Set the figure patch face and edge
        colors.  This is useful because some of the GUIs have a gray figure
        face color background and you'll probably want to override this on
        hardcopy.

        orientation - only currently applies to PostScript printing.

	dpi - ignored
        """
        self.figure.dpi.set(72)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)        
	width, height = self.figure.get_size_inches()

	file = PdfFile(width, height, filename)
        renderer = RendererPdf(file)
	self.figure.draw(renderer)
	file.close()

class FigureManagerPdf(FigureManagerBase):
    pass

FigureManager = FigureManagerPdf
