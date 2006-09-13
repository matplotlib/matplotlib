# -*- coding: iso-8859-1 -*-
"""
A PDF matplotlib backend (not yet complete)
Author: Jouni K Seppänen <jks@iki.fi>
"""
from __future__ import division

import md5
import os
import re
import sys
import time
import zlib

from cStringIO import StringIO
from datetime import datetime
from math import ceil, cos, floor, pi, sin

from matplotlib import __version__, rcParams, agg
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import Bunch, enumerate, is_string_like
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.ft2font import FT2Font, FIXED_WIDTH, ITALIC, LOAD_NO_SCALE
from matplotlib.mathtext import math_parse_s_pdf
from matplotlib.numerix import Float32, UInt8, fromstring, arange, infinity, isnan, asarray
from matplotlib.transforms import Bbox

# Overview
#
# The low-level knowledge about pdf syntax lies mainly in the pdfRepr
# function and the classes Reference, Name, Operator, and Stream.  The
# PdfFile class knows about the overall structure of pdf documents.
# It provides a "write" method for writing arbitrary strings in the
# file, and an "output" method that passes objects through the pdfRepr
# function before writing them in the file.  The output method is
# called by the RendererPdf class, which contains the various draw_foo
# methods.  RendererPdf contains a GraphicsContextPdf instance, and
# each draw_foo calls self.check_gc before outputting commands.  This
# method checks whether the pdf graphics state needs to be modified
# and outputs the necessary commands.  GraphicsContextPdf represents
# the graphics state, and its "delta" method returns the commands that
# modify the state.
#
# Some tricky points: 
#
# 1. The clip rectangle (which could in pdf be an arbitrary path, not
# necessarily a rectangle) can only be widened by popping from the
# state stack.  Thus the state must be pushed onto the stack before
# narrowing the rectangle.  This is taken care of by
# GraphicsContextPdf.
#
# 2. Sometimes it is necessary to refer to something (e.g. font,
# image, or extended graphics state, which contains the alpha value)
# in the page stream by a name that needs to be defined outside the
# stream.  PdfFile provides the methods fontName, imageObject, and
# alphaState for this purpose.  The implementations of these methods
# should perhaps be generalized.

# TODOs:
#
# * the alpha channel of images
# * image compression could be improved (PDF supports png-like compression) 
# * encoding of fonts, including mathtext fonts and unicode support
# * Type 1 and Base-14 font support (i.e., "pdf.use_afm") 
# * TTF support has lots of small TODOs, e.g. how do you know if a font  
#   is serif/sans-serif, or symbolic/non-symbolic? 
# * draw_markers, draw_line_collection, etc. 
# * use_tex 

def fill(strings, linelen=75):
    """Make one string from sequence of strings, with whitespace
    in between. The whitespace is chosen to form lines of at most
    linelen characters, if possible."""

    s, strings = [strings[0]], strings[1:]
    while strings:
	if len(s[-1]) + len(strings[0]) < linelen:
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
	if isnan(obj) or obj in (-infinity, infinity):
	    raise ValueError, "Can only output finite numbers in PDF"
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

    # A date.
    elif isinstance(obj, datetime):
	r = obj.strftime('D:%Y%m%d%H%M%S')
	if time.daylight: z = time.altzone
	else: z = time.timezone
	if z == 0: r += 'Z'
	elif z < 0: r += "+%02d'%02d'" % ((-z)//3600, (-z)%3600)
	else: r += "-%02d'%02d'" % (z//3600, z%3600)
	return pdfRepr(r)

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

    def __repr__(self):
	return "<Reference %d>" % self.id

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
	    self.name = re.sub(r'[^!-~]', Name.hexify, name)

    def __repr__(self):
	return "<Name %s>" % self.name

    def hexify(match):
	return '#%02x' % ord(match.group())
    hexify = staticmethod(hexify)

    def pdfRepr(self):
	return '/' + self.name

class Operator:
    """PDF operator object."""

    def __init__(self, op):
	self.op = op

    def __repr__(self):
	return '<Operator %s>' % self.op

    def pdfRepr(self):
	return self.op

# PDF operators (not an exhaustive list)
_pdfops = dict(close_fill_stroke='b', fill_stroke='B', fill='f',
	       closepath='h', close_stroke='s', stroke='S', endpath='n',
	       begin_text='BT', end_text='ET',
	       curveto='c', rectangle='re', lineto='l', moveto='m',
	       concat_matrix='cm',
	       use_xobject='Do',
	       setgray_stroke='G', setgray_nonstroke='g',
	       setrgb_stroke='RG', setrgb_nonstroke='rg',
	       setcolorspace_stroke='CS', setcolorspace_nonstroke='cs',
	       setcolor_stroke='SCN', setcolor_nonstroke='scn',
	       setdash='d', setlinejoin='j', setlinecap='J', setgstate='gs',
	       gsave='q', grestore='Q',
	       textpos='Td', selectfont='Tf', textmatrix='Tm',
	       show='Tj', showkern='TJ',
	       setlinewidth='w', clip='W')

Op = Bunch(**dict([(name, Operator(value)) 
		   for name, value in _pdfops.items()]))

class Stream:
    """PDF stream object.

    This has no pdfRepr method. Instead, call begin(), then output the
    contents of the stream by calling write(), and finally call end().
    """

    def __init__(self, id, len, file, extra=None):
	"""id: object id of stream; len: an unused Reference object for the
	length of the stream, or None (to use a memory buffer); file:
	a PdfFile; extra: a dictionary of extra key-value pairs to
	include in the stream header """
	self.id = id		# object id
	self.len = len		# id of length object
	self.pdfFile = file
	self.file = file.fh	# file to which the stream is written
	self.compressobj = None	# compression object
	if extra is None: self.extra = dict()
	else: self.extra = extra

	self.pdfFile.recordXref(self.id)
	if rcParams['pdf.compression']:
	    self.compressobj = zlib.compressobj(rcParams['pdf.compression'])
	if self.len is None:
	    self.file = StringIO()
	else: 
	    self._writeHeader()
	    self.pos = self.file.tell()

    def _writeHeader(self):
	write = self.file.write
	write("%d 0 obj\n" % self.id)
	dict = self.extra
	dict['Length'] = self.len
	if rcParams['pdf.compression']:
	    dict['Filter'] = Name('FlateDecode')

	write(pdfRepr(dict))
	write("\nstream\n")

    def end(self):
	"""Finalize stream."""

	self._flush()
	if self.len is None:
	    contents = self.file.getvalue()
	    self.len = len(contents)
	    self.file = self.pdfFile.fh
	    self._writeHeader()
	    self.file.write(contents)
	    self.file.write("\nendstream\nendobj\n")
	else:
	    length = self.file.tell() - self.pos
	    self.file.write("\nendstream\nendobj\n")
	    self.pdfFile.writeObject(self.len, length)

    def write(self, data):
	"""Write some data on the stream."""

	if self.compressobj is None:
	    self.file.write(data)
	else:
	    compressed = self.compressobj.compress(data)
	    self.file.write(compressed)

    def _flush(self):
	"""Flush the compression object."""

	if self.compressobj is not None:
	    compressed = self.compressobj.flush()
	    self.file.write(compressed)
	    self.compressobj = None

class PdfFile:
    """PDF file with one page."""

    def __init__(self, width, height, dpi, filename):
	self.nextObject = 1	# next free object id
	self.xrefTable = [ [0, 65535, 'the zero object'] ]
	self.dpi = dpi
	fh = file(filename, 'wb')
	self.fh = fh
	self.currentstream = None # stream object to write to, if any
	fh.write("%PDF-1.4\n")	  # 1.4 is the first version to have alpha
	# Output some eight-bit chars as a comment so various utilities
	# recognize the file as binary by looking at the first few
	# lines (see note in section 3.4.1 of the PDF reference).
	fh.write("%\254\334 \253\272\n")

	self.rootObject = self.reserveObject('root')
	self.infoObject = self.reserveObject('info')
	pagesObject = self.reserveObject('pages')
	thePageObject = self.reserveObject('page 0')
	contentObject = self.reserveObject('contents of page 0')
	self.fontObject = self.reserveObject('fonts')
	self.alphaStateObject = self.reserveObject('extended graphics states')
	self.hatchObject = self.reserveObject('tiling patterns')
	self.XObjectObject = self.reserveObject('external objects')
	resourceObject = self.reserveObject('resources')

	root = { 'Type': Name('Catalog'),
		 'Pages': pagesObject }
	self.writeObject(self.rootObject, root)

	info = { 'Creator': 'matplotlib ' + __version__ \
		 + ', http://matplotlib.sf.net',
		 'Producer': 'matplotlib pdf backend',
		 'CreationDate': datetime.today() }

	# Possible TODO: Title, Author, Subject, Keywords
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
	self.hatchPatterns = {}
	self.nextHatch = 1

	self.images = {}
	self.nextImage = 1

	self.markers = {}
	self.nextMarker = 1

	# The PDF spec recommends to include every procset
	procsets = [ Name(x)
		     for x in "PDF Text ImageB ImageC ImageI".split() ]

	# Write resource dictionary.
	# Possibly TODO: more general ExtGState (graphics state dictionaries)
	#                ColorSpace Pattern Shading Properties
	resources = { 'Font': self.fontObject,
		      'XObject': self.XObjectObject,
		      'ExtGState': self.alphaStateObject,
		      'Pattern': self.hatchObject,
		      'ProcSet': procsets }
	self.writeObject(resourceObject, resources)

	# Start the content stream of the page
	self.beginStream(contentObject.id,
			 self.reserveObject('length of content stream'))

    def close(self):
	# End the content stream and write out the various deferred
	# objects
	self.endStream()
	self.writeFonts()
	self.writeObject(self.alphaStateObject,
			 dict([(val[0], val[1])
			       for val in self.alphaStates.values()]))
	self.writeHatches()
	xobjects = dict(self.images.values())
	xobjects.update([(name, value[0]) for (name, value) in self.markers.items()])
	self.writeObject(self.XObjectObject, xobjects)
	self.writeImages()
	self.writeMarkers()
	self.writeXref()
	self.writeTrailer()
	self.fh.close()

    def write(self, data):
	if self.currentstream is None:
	    self.fh.write(data)
	else:
	    self.currentstream.write(data)

    def output(self, *data):
	self.write(fill(map(pdfRepr, data)))
	self.write('\n')

    def beginStream(self, id, len, extra=None):
	assert self.currentstream is None
	self.currentstream = Stream(id, len, self, extra)

    def endStream(self):
	self.currentstream.end()
	self.currentstream = None

    # These fonts do not need to be embedded; every PDF viewing
    # application is required to have them.
    base14 = [ 'Times-Roman', 'Times-Bold', 'Times-Italic',
	       'Times-BoldItalic', 'Symbol', 'ZapfDingbats' ] + \
	     [ prefix + postfix
	       for prefix in 'Helvetica', 'Courier'
	       for postfix in '', '-Bold', '-Oblique', '-BoldOblique' ]

    def fontName(self, fontprop):
	if is_string_like(fontprop):
	    filename = fontprop
	else:
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
		fontdictObject = self.embedTTF(filename)
	    fonts[Fx] = fontdictObject
	    #print >>sys.stderr, filename

	self.writeObject(self.fontObject, fonts)

    def embedTTF(self, filename):
	"""Embed the TTF font from the named file into the document."""

	font = FT2Font(filename)

	def cvt(length, upe=font.units_per_EM, nearest=True):
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
	firstchar, lastchar = chars[0], chars[-1]

	# Get widths
	widths = [ cvt(font.load_char(i, flags=LOAD_NO_SCALE).horiAdvance)
		   for i in range(firstchar, lastchar+1) ]
	# Remove redundant widths from end and beginning; doing the
	# end first helps reduce LastChar, which apparently needs to
	# be less than 256 or acroread complains.
	missingwidth = widths[-1]
	while len(widths)>1 and widths[-1] == missingwidth:
	    lastchar -= 1
	    widths.pop()
	while len(widths)>1 and widths[0] == missingwidth:
	    firstchar += 1
	    widths.pop(0)

	widthsObject = self.reserveObject('font widths')
	fontdescObject = self.reserveObject('font descriptor')
 	fontdict = { 'Type': Name('Font'),
		     'Subtype': Name('TrueType'),
		     'Encoding': Name('WinAnsiEncoding'), # ???
		     'BaseFont': ps_name,
		     'FirstChar': firstchar,
		     'LastChar': lastchar,
		     'Widths': widthsObject,
		     'FontDescriptor': fontdescObject }
	# TODO: Encoding?

	flags = 0
	symbolic = False #ps_name.name in ('Cmsy10', 'Cmmi10', 'Cmex10')
	if ff & FIXED_WIDTH: flags |= 1 << 0
	if 0: flags |= 1 << 1 # TODO: serif
	if symbolic: flags |= 1 << 2
	else: flags |= 1 << 5
	if sf & ITALIC: flags |= 1 << 6
	if 0: flags |= 1 << 16 # TODO: all caps
	if 0: flags |= 1 << 17 # TODO: small caps
	if 0: flags |= 1 << 18 # TODO: force bold

	descriptor = {
	    'Type': Name('FontDescriptor'),
	    'FontName': ps_name,
	    'Flags': flags,
	    'FontBBox': [ cvt(x, nearest=False) for x in font.bbox ],
	    'Ascent': cvt(font.ascender, nearest=False),
	    'Descent': cvt(font.descender, nearest=False),
	    'CapHeight': cvt(pclt['capHeight'], nearest=False),
	    'XHeight': cvt(pclt['xHeight']),
	    'ItalicAngle': post['italicAngle'][1], # ???
	    'FontFile2': self.reserveObject('font file'),
	    'MaxWidth': max(widths+[missingwidth]),
	    'MissingWidth': missingwidth,
	    'StemV': 0 # ???
	    }

	# Other FontDescriptor keys include:
	# /FontFamily /Times (optional)
	# /FontStretch /Normal (optional)
	# /FontFile (stream for type 1 font)
	# /CharSet (used when subsetting type1 fonts)

	# Make an Identity-H encoded CID font for CM fonts? (Doesn't quite work)
	if False:
	    del fontdict['Widths'], fontdict['FontDescriptor'], \
		fontdict['LastChar'], fontdict['FirstChar']

	    fontdict['Subtype'] = Name('Type0')
	    fontdict['Encoding'] = Name('Identity-H')
	    fontdict2Object = self.reserveObject('descendant font')
	    fontdict['DescendantFonts'] = [ fontdict2Object ]
	    # TODO: fontdict['ToUnicode']
	    fontdict2 = { 'Type': Name('Font'),
			  'Subtype': Name('CIDFontType2'),
			  'BaseFont': ps_name,
			  'W': widthsObject,
			  'DW': missingwidth,
			  'CIDSystemInfo': { 'Registry': 'Adobe', 
					     'Ordering': 'Identity', 
					     'Supplement': 0 },
			  'FontDescriptor': fontdescObject }
	    self.writeObject(fontdict2Object, fontdict2)

	    widths = [ firstchar, widths ]

	fontdictObject = self.reserveObject('font dictionary')
	length1Object = self.reserveObject('decoded length of a font')
	self.writeObject(fontdictObject, fontdict)
	self.writeObject(widthsObject, widths)
	self.writeObject(fontdescObject, descriptor)
	self.beginStream(descriptor['FontFile2'].id,
			 self.reserveObject('length of font stream'),
			 {'Length1': length1Object})
	fontfile = open(filename, 'rb')
	length1 = 0
	while True:
	    data = fontfile.read(4096)
	    if not data: break
	    length1 += len(data)
	    self.currentstream.write(data)
	self.endStream()
	self.writeObject(length1Object, length1)

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

    def hatchPattern(self, lst):
	pattern = self.hatchPatterns.get(lst, None)
	if pattern is not None:
	    return pattern[0]

	name = Name('H%d' % self.nextHatch)
	self.nextHatch += 1
	self.hatchPatterns[lst] = name
	return name

    def writeHatches(self):
	hatchDict = dict()
	sidelen = 144.0
	density = 24.0
	for lst, name in self.hatchPatterns.items():
	    ob = self.reserveObject('hatch pattern')
	    hatchDict[name] = ob
	    res = { 'Procsets': 
		    [ Name(x) for x in "PDF Text ImageB ImageC ImageI".split() ] }
	    self.beginStream(
		ob.id, None, 
		{ 'Type': Name('Pattern'),
		  'PatternType': 1, 'PaintType': 1, 'TilingType': 1,
		  'BBox': [0, 0, sidelen, sidelen],
		  'XStep': sidelen, 'YStep': sidelen,
		  'Resources': res })

	    # lst is a tuple of stroke color, fill color, 
	    # number of - lines, number of / lines, 
	    # number of | lines, number of \ lines
	    rgb = lst[0]
	    self.output(rgb[0], rgb[1], rgb[2], Op.setrgb_stroke)
	    if lst[1] is not None:
		rgb = lst[1]
		self.output(rgb[0], rgb[1], rgb[2], Op.setrgb_nonstroke,
			    0, 0, sidelen, sidelen, Op.rectangle,
			    Op.fill)
	    if lst[2]:		# -
		for j in arange(0.0, sidelen, density/lst[2]):
		    self.output(0, j, Op.moveto,
				sidelen, j, Op.lineto)
	    if lst[3]:		# /
		for j in arange(0.0, sidelen, density/lst[3]):
		    self.output(0, j, Op.moveto,
				sidelen-j, sidelen, Op.lineto,
				sidelen-j, 0, Op.moveto,
				sidelen, j, Op.lineto)
	    if lst[4]:		# |
		for j in arange(0.0, sidelen, density/lst[4]):
		    self.output(j, 0, Op.moveto,
				j, sidelen, Op.lineto)
	    if lst[5]:		# \
		for j in arange(sidelen, 0.0, -density/lst[5]):
		    self.output(sidelen, j, Op.moveto,
				j, sidelen, Op.lineto,
				j, 0, Op.moveto,
				0, j, Op.lineto)
	    self.output(Op.stroke)

	    self.endStream()
	self.writeObject(self.hatchObject, hatchDict)

    def imageObject(self, image):
	"""Return name of an image XObject representing the given image."""

	pair = self.images.get(image, None)
	if pair is not None:
	    return pair[0]

	name = Name('I%d' % self.nextImage)
	ob = self.reserveObject('image %d' % self.nextImage)
	self.nextImage += 1
	self.images[image] = (name, ob)
	return name

    ## These two from backend_ps.py
    ## TODO: alpha (SMask, p. 518 of pdf spec)

    def _rgb(self, im):
        h,w,s = im.as_rgba_str()

        rgba = fromstring(s, UInt8)
        rgba.shape = (h, w, 4)
        rgb = rgba[:,:,:3]
        return h, w, rgb.tostring()

    def _gray(self, im, rc=0.3, gc=0.59, bc=0.11):
        rgbat = im.as_rgba_str()
        rgba = fromstring(rgbat[2], UInt8)
        rgba.shape = (rgbat[0], rgbat[1], 4)
        rgba_f = rgba.astype(Float32)
        r = rgba_f[:,:,0]
        g = rgba_f[:,:,1]
        b = rgba_f[:,:,2]
        gray = (r*rc + g*gc + b*bc).astype(UInt8)
        return rgbat[0], rgbat[1], gray.tostring()

    def writeImages(self):
	for img, pair in self.images.items():
	    img.flipud_out()
	    if img.is_grayscale:
		height, width, data = self._gray(img)
		colorspace = Name('DeviceGray')
	    else:
		height, width, data = self._rgb(img)
		colorspace = Name('DeviceRGB')

	    self.beginStream(
		pair[1].id,
		self.reserveObject('length of image stream'), 
		{'Type': Name('XObject'), 'Subtype': Name('Image'),
		 'Width': width, 'Height': height,
		 'ColorSpace': colorspace, 'BitsPerComponent': 8 })
	    self.currentstream.write(data) # TODO: predictors (i.e., output png)
	    self.endStream()

	    img.flipud_out()

    def markerObject(self, path, fillp, lw):
	"""Return name of a marker XObject representing the given path."""

	name = Name('M%d' % self.nextMarker)
	ob = self.reserveObject('marker %d' % self.nextMarker)
	self.nextMarker += 1
	self.markers[name] = (ob, path, fillp, lw)
	return name

    def writeMarkers(self):
	for name, tuple in self.markers.items():
	    object, path, fillp, lw = tuple
	    self.beginStream(
		object.id, None, 
		{'Type': Name('XObject'), 'Subtype': Name('Form'),
		 'BBox': self.pathBbox(path, lw) })
	    self.writePath(path, fillp)
	    self.endStream()

    def pathBbox(path, lw):
	path.rewind(0)
	x, y = [], []
	while True:
	    code, xp, yp = path.vertex()
	    if code & agg.path_cmd_mask in \
		    (agg.path_cmd_move_to, agg.path_cmd_line_to):
		x.append(xp)
		y.append(yp)
	    elif code == agg.path_cmd_stop:
		break
	return min(x)-lw, min(y)-lw, max(x)+lw, max(y)+lw
    pathBbox = staticmethod(pathBbox)

    def writePath(self, path, fillp):
	path.rewind(0)
	while True:
	    code, xp, yp = path.vertex()
	    code = code & agg.path_cmd_mask
	    if code == agg.path_cmd_stop:
                break
            elif code == agg.path_cmd_move_to:
		self.output(xp, yp, Op.moveto)
            elif code == agg.path_cmd_line_to:
		self.output(xp, yp, Op.lineto)
            elif code == agg.path_cmd_curve3:
                pass
            elif code == agg.path_cmd_curve4:
                pass
            elif code == agg.path_cmd_end_poly:
		self.output(Op.closepath)
            else:
                print >>sys.stderr, "writePath", code, xp, yp
	if fillp:
	    self.output(Op.fill_stroke)
	else:
	    self.output(Op.stroke)

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
	# Could add 'ID'
	self.write("\nstartxref\n%d\n%%%%EOF\n" % self.startxref)

class RendererPdf(RendererBase):

    def __init__(self, file):
	self.file = file
	self.gc = self.new_gc()
	self.fonts = {}
	self.dpi_factor = 72.0/file.dpi

    def points_to_pixels(self, points):
	return points / self.dpi_factor

    def finalize(self):
	self.gc.finalize()

    def check_gc(self, gc, fillcolor=None):
	orig_fill = gc._fillcolor
	gc._fillcolor = fillcolor

	delta = self.gc.delta(gc)
	if delta: self.file.output(*delta)

	# Restore gc to avoid unwanted side effects
	gc._fillcolor = orig_fill

    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, 
		 angle1, angle2, rotation):
        """
        Draw an arc using GraphicsContext instance gcEdge, centered at x,y,
        with width and height and angles from 0.0 to 360.0
        0 degrees is at 3-o'clock, rotated by `rotation` degrees
        positive angles are anti-clockwise

        If the color rgbFace is not None, fill the arc with it.
        """
	# source: agg_bezier_arc.cpp in agg23

	x *= self.dpi_factor
	y *= self.dpi_factor
	width *= self.dpi_factor
	height *= self.dpi_factor
	
	def arc_to_bezier(cx, cy, rx, ry, angle1, sweep):
	    halfsweep = sweep / 2.0
	    x0, y0 = cos(halfsweep), sin(halfsweep)
	    tx = (1.0 - x0) * 4.0/3.0;
	    ty = y0 - tx * x0 / y0;
	    px =  x0, x0+tx, x0+tx, x0
	    py = -y0,   -ty,    ty, y0
	    sn, cs = sin(angle1 + halfsweep), cos(angle1 + halfsweep)
	    result = [ (cx + rx * (pxi * cs - pyi * sn),
			cy + ry * (pxi * sn + pyi * cs))
		       for pxi, pyi in zip(px, py) ]
	    return reduce(lambda x, y: x + y, result)

	epsilon = 0.01
	angle1 *= pi/180.0
	angle2 *= pi/180.0
	rotation *= pi/180.0
	sweep = angle2 - angle1
	angle1 = angle1 % (2*pi)
	sweep = min(max(-2*pi, sweep), 2*pi)

	if sweep < 0.0:
	    sweep, angle1, angle2 = -sweep, angle2, angle1
	bp = [ rotation + pi/2.0 * i 
	       for i in range(4) if pi/2.0 * i < sweep-epsilon ]
	bp.append(rotation + sweep)
	subarcs = [ arc_to_bezier(x, y, width/2.0, height/2.0,
				  bp[i], bp[i+1]-bp[i]) 
		    for i in range(len(bp)-1) ]

	self.check_gc(gcEdge, rgbFace)
	self.file.output(subarcs[0][0], subarcs[0][1], Op.moveto)
	for arc in subarcs:
	    self.file.output(*(arc[2:] + (Op.curveto,)))

	self.file.output(self.gc.close_and_paint())

    def draw_image(self, x, y, im, bbox):
        #print >>sys.stderr, "draw_image called"

	gc = self.new_gc()
	gc.set_clip_rectangle(bbox.get_bounds())
	self.check_gc(gc)

	h, w = im.get_size_out()
	d = self.dpi_factor
	imob = self.file.imageObject(im)
	self.file.output(Op.gsave, d*w, 0, 0, d*h, d*x, d*y, Op.concat_matrix,
			 imob, Op.use_xobject, Op.grestore)

    def draw_line(self, gc, x1, y1, x2, y2):
	if isnan(x1) or isnan(x2) or isnan(y1) or isnan(y2):
	    return
	d = self.dpi_factor
	self.check_gc(gc)
	self.file.output(d*x1, d*y1, Op.moveto,
			 d*x2, d*y2, Op.lineto, self.gc.paint())

    def draw_lines(self, gc, x, y, transform=None):
	d = self.dpi_factor
	self.check_gc(gc)
	if transform is not None:
	    x, y = transform.seq_x_y(x, y)
	nan_at = isnan(x) | isnan(y)
	next_op = Op.moveto
	for i in range(len(x)):
	    if nan_at[i]:
		next_op = Op.moveto
	    else:
		self.file.output(d*x[i], d*y[i], next_op)
		next_op = Op.lineto
	self.file.output(self.gc.paint())

    def draw_point(self, gc, x, y):
        print >>sys.stderr, "draw_point called"

	d = self.dpi_factor
	self.check_gc(gc, gc._rgb)
	self.file.output(d*x, d*y, d, d,
			 Op.rectangle, Op.fill_stroke)

    def draw_polygon(self, gcEdge, rgbFace, points):
	# Optimization for axis-aligned rectangles
	if len(points) == 4:
	    if points[0][0] == points[1][0] and points[1][1] == points[2][1] and \
	       points[2][0] == points[3][0] and points[3][1] == points[0][1]:
		self.draw_rectangle(gcEdge, rgbFace, 
				    min(points[0][0], points[2][0]),
				    min(points[1][1], points[3][1]),
				    abs(points[2][0] - points[0][0]),
				    abs(points[3][1] - points[1][1]))
		return
	    elif points[0][1] == points[1][1] and points[1][0] == points[2][0] and \
	         points[2][1] == points[3][1] and points[3][0] == points[0][0]:
		self.draw_rectangle(gcEdge, rgbFace, 
				    min(points[1][0], points[3][0]),
				    min(points[2][1], points[0][1]),
				    abs(points[1][0] - points[3][0]),
				    abs(points[2][1] - points[0][1]))
		return

	self.check_gc(gcEdge, rgbFace)
	d = self.dpi_factor
	self.file.output(d*points[0][0], d*points[0][1], Op.moveto)
	for x,y in points[1:]:
	    self.file.output(d*x, d*y, Op.lineto)
	self.file.output(self.gc.close_and_paint())

    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
	self.check_gc(gcEdge, rgbFace)
	d = self.dpi_factor
	self.file.output(d*x, d*y, d*width, d*height, Op.rectangle)
	self.file.output(self.gc.paint())

    def draw_markers(self, gc, path, rgbFace, x, y, trans):
	self.check_gc(gc, rgbFace)
	fillp = rgbFace is not None
	marker = self.file.markerObject(path, fillp, self.gc._linewidth)
	x, y = trans.numerix_x_y(asarray(x), asarray(y))
	x, y = self.dpi_factor * x, self.dpi_factor * y
	nan_at = isnan(x) | isnan(y)

	self.file.output(Op.gsave)
	ox, oy = 0, 0
	for i in range(len(x)):
	    if nan_at[i]: continue
	    dx, dy, ox, oy = x[i]-ox, y[i]-oy, x[i], y[i]
	    self.file.output(1, 0, 0, 1, dx, dy, 
			     Op.concat_matrix,
			     marker, Op.use_xobject)
	self.file.output(Op.grestore)

    def _setup_textpos(self, x, y, angle, oldx=0, oldy=0, oldangle=0):
	d = self.dpi_factor
	if angle == oldangle == 0:
	    self.file.output(d*(x-oldx), d*(y-oldy), Op.textpos)
	else:
	    angle = angle / 180.0 * pi
	    self.file.output( cos(angle), sin(angle),
			     -sin(angle), cos(angle),
			      d*x,        d*y,         Op.textmatrix)

    def draw_mathtext(self, gc, x, y, s, prop, angle):
	# TODO: fix positioning and encoding
	fontsize = prop.get_size_in_points()
	width, height, pswriter = math_parse_s_pdf(s, self.file.dpi, fontsize)

	self.check_gc(gc, gc._rgb)
	self.file.output(Op.begin_text)
	prev_font = None, None
	oldx, oldy = 0, 0
	for ox, oy, fontname, fontsize, glyph in pswriter:
	    #print ox, oy, glyph
	    fontname = fontname.lower()
	    a = angle / 180.0 * pi
	    newx = x + cos(a)*ox - sin(a)*oy
	    newy = y + sin(a)*ox + cos(a)*oy
	    self._setup_textpos(newx, newy, angle, oldx, oldy)
	    oldx, oldy = newx, newy
	    if (fontname, fontsize) != prev_font:
		self.file.output(self.file.fontName(fontname), fontsize, 
				 Op.selectfont)
		prev_font = fontname, fontsize

	    #if fontname.endswith('cmsy10.ttf') or \
	    #fontname.endswith('cmmi10.ttf') or \
	    #fontname.endswith('cmex10.ttf'):
	    #	string = '\0' + chr(glyph)

	    string = chr(glyph)
	    self.file.output(string, Op.show)
	self.file.output(Op.end_text)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
	# TODO: combine consecutive texts into one BT/ET delimited section
	#       unicode
	if isinstance(s, unicode):
	    s = s.encode('ascii', 'replace')
	if ismath: return self.draw_mathtext(gc, x, y, s, prop, angle)
	self.check_gc(gc, gc._rgb)

	font = self._get_font_ttf(prop)
	font.set_text(s, 0.0)
	y += font.get_descent() / 64.0

	self.file.output(Op.begin_text, 
			 self.file.fontName(prop),
			 prop.get_size_in_points(),
			 Op.selectfont)

	self._setup_textpos(x, y, angle)
	self.file.output(s, Op.show, Op.end_text)

    def get_text_width_height(self, s, prop, ismath):

	if ismath:
	    fontsize = prop.get_size_in_points()
	    w, h, pswriter = math_parse_s_pdf(s, self.file.dpi, fontsize)
	else:
	    font = self._get_font_ttf(prop)
	    font.set_text(s, 0.0)
	    w, h = font.get_width_height()
	    factor = 1.0/(self.dpi_factor*64.0)
	    w *= factor
	    h *= factor
	return w, h

    def _get_font_ttf(self, prop):
	font = self.fonts.get(prop)
	if font is None:
	    font = FT2Font(fontManager.findfont(prop))
	    self.fonts[prop] = font
	font.clear()
	font.set_size(prop.get_size_in_points(), 72.0)
	return font

    def flipy(self):
        return False

    def get_canvas_width_height(self):
	d = self.dpi_factor/72.0
        return d*self.file.width, d*self.file.height

    def new_gc(self):
        return GraphicsContextPdf(self.file)

    def points_to_pixels(self, points):
        return points


class GraphicsContextPdf(GraphicsContextBase):

    def __init__(self, file):
	GraphicsContextBase.__init__(self)
	self._fillcolor = (0.0, 0.0, 0.0)
	self.file = file
	self.parent = None

    def __repr__(self):
	d = dict(self.__dict__)
	del d['file']
	del d['parent']
	return `d`

    def _strokep(self):
	return self._linewidth > 0 and self._alpha > 0

    def _fillp(self):
	return self._fillcolor is not None or self._hatch

    def close_and_paint(self):
	if self._strokep():
	    if self._fillp(): 
		return Op.close_fill_stroke
	    else:
		return Op.close_stroke
	else:
	    if self._fillp():
		return Op.fill
	    else:
		return Op.endpath

    def paint(self):
	if self._strokep():
	    if self._fillp(): 
		return Op.fill_stroke
	    else:
		return Op.stroke
	else:
	    if self._fillp():
		return Op.fill
	    else:
		return Op.endpath

    capstyles = { 'butt': 0, 'round': 1, 'projecting': 2 }
    joinstyles = { 'miter': 0, 'round': 1, 'bevel': 2 }

    def capstyle_cmd(self, style):
	return [self.capstyles[style], Op.setlinecap]

    def joinstyle_cmd(self, style):
	return [self.joinstyles[style], Op.setlinejoin]

    def linewidth_cmd(self, width):
	return [width, Op.setlinewidth]

    def dash_cmd(self, dashes):
	offset, dash = dashes
	if dash is None: 
	    dash = []
	    offset = 0
	return [list(dash), offset, Op.setdash]

    def alpha_cmd(self, alpha):
	name = self.file.alphaState(alpha)
	return [name, Op.setgstate]

    def hatch_cmd(self, hatch):
	if not hatch:
	    if self._fillcolor:
		return self.fillcolor_cmd(self._fillcolor)
	    else:
		return [Name('DeviceRGB'), Op.setcolorspace_nonstroke]
	else:
	    hatch = hatch.lower()
	    lst = ( self._rgb,
		    self._fillcolor,
		    hatch.count('-') + hatch.count('+'),
		    hatch.count('/') + hatch.count('x'),
		    hatch.count('|') + hatch.count('+'),
		    hatch.count('\\') + hatch.count('x') )
	    name = self.file.hatchPattern(lst)
	    return [Name('Pattern'), Op.setcolorspace_nonstroke,
		    name, Op.setcolor_nonstroke]

    def rgb_cmd(self, rgb):
	if rgb[0] == rgb[1] == rgb[2]:
	    return [rgb[0], Op.setgray_stroke]
	else:
	    return list(rgb) + [Op.setrgb_stroke] 

    def fillcolor_cmd(self, rgb):
	if rgb is None:
	    return []
	elif rgb[0] == rgb[1] == rgb[2]:
	    return [rgb[0], Op.setgray_nonstroke]
	else:
	    return list(rgb) + [Op.setrgb_nonstroke] 

    def push(self):
	parent = GraphicsContextPdf(self.file)
	parent.copy_properties(self)
	parent.parent = self.parent
	self.parent = parent
	return [Op.gsave]

    def pop(self):
	assert self.parent is not None
	self.copy_properties(self.parent)
	self.parent = self.parent.parent
	return [Op.grestore]

    def cliprect_cmd(self, cliprect):
	"""Set clip rectangle. Calls self.pop() and self.push()."""
	d = 72.0/self.file.dpi

	cmds = []
	while self._cliprect != cliprect and self.parent is not None:
	    cmds.extend(self.pop())
	if self._cliprect != cliprect:
	    cmds.extend(self.push() + 
			[d*t for t in cliprect] + 
			[Op.rectangle, Op.clip, Op.endpath])
	return cmds

    commands = (
	('_cliprect', cliprect_cmd), # must come first since may pop
	('_alpha', alpha_cmd),
	('_capstyle', capstyle_cmd),
	('_fillcolor', fillcolor_cmd),
	('_joinstyle', joinstyle_cmd),
	('_linewidth', linewidth_cmd),
	('_dashes', dash_cmd),
	('_rgb', rgb_cmd),
	('_hatch', hatch_cmd),	# must come after fillcolor and rgb
	)

    # TODO: _linestyle

    def copy_properties(self, other):
	"""Copy properties of other into self."""
	GraphicsContextBase.copy_properties(self, other)
	self._fillcolor = other._fillcolor

    def delta(self, other):
	"""Copy properties of other into self and return PDF commands 
	needed to transform self into other.
	"""
	cmds = []
	for param, cmd in self.commands:
	    if getattr(self, param) != getattr(other, param):
		cmds.extend(cmd(self, getattr(other, param)))
		setattr(self, param, getattr(other, param))
	return cmds

    def finalize(self):
	"""Make sure every pushed graphics state is popped."""
	cmds = []
	while self.parent is not None:
	    cmds.extend(self.pop())
	return cmds

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
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
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

    def print_figure(self, filename, dpi=None, facecolor='w', edgecolor='w',
                     orientation='portrait', **kwargs):
        """
        Render the figure to hardcopy. Set the figure patch face and edge
        colors.  This is useful because some of the GUIs have a gray figure
        face color background and you'll probably want to override this on
        hardcopy.

        orientation - only currently applies to PostScript printing.

	dpi - used for images
        """
        if dpi is None:
            dpi = rcParams['savefig.dpi']
        self.figure.dpi.set(dpi)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)
	width, height = self.figure.get_size_inches()

	basename, ext = os.path.splitext(filename)
	if ext == '': 
	    filename += '.pdf'

	file = PdfFile(width, height, dpi, filename)
        renderer = RendererPdf(file)
	self.figure.draw(renderer)
	renderer.finalize()
	file.close()

class FigureManagerPdf(FigureManagerBase):
    pass

FigureManager = FigureManagerPdf
