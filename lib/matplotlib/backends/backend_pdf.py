# -*- coding: iso-8859-1 -*-
"""
A PDF matplotlib backend (not yet complete)
Author: Jouni K Seppänen <jks@iki.fi>
"""
from __future__ import division

import os
import re
import sys
import time
import warnings
import zlib

import numpy as npy

from cStringIO import StringIO
from datetime import datetime
from math import ceil, cos, floor, pi, sin
from sets import Set

import matplotlib
from matplotlib import __version__, rcParams, agg, get_data_path
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase
from matplotlib.cbook import Bunch, enumerate, is_string_like, reverse_dict, \
    get_realpath_and_stat, is_writable_file_like, maxdict
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont, is_opentype_cff_font
from matplotlib.afm import AFM
import matplotlib.type1font as type1font
import matplotlib.dviread as dviread
from matplotlib.ft2font import FT2Font, FIXED_WIDTH, ITALIC, LOAD_NO_SCALE, \
    LOAD_NO_HINTING, KERNING_UNFITTED
from matplotlib.mathtext import MathTextParser
from matplotlib.transforms import Bbox
from matplotlib import ttconv

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

# Add "pdf.use14corefonts: True" in your configuration file to use only
# the 14 PDF core fonts. These fonts do not need to be embedded; every
# PDF viewing application is required to have them. This results in very
# light PDF files you can use directly in LaTeX or ConTeXt documents
# generated with pdfTeX, without any conversion.

# These fonts are: Helvetica, Helvetica-Bold, Helvetica-Oblique,
# Helvetica-BoldOblique, Courier, Courier-Bold, Courier-Oblique,
# Courier-BoldOblique, Times-Roman, Times-Bold, Times-Italic,
# Times-BoldItalic, Symbol, ZapfDingbats.
#
# Some tricky points:
#
# 1. The clip path can only be widened by popping from the state
# stack.  Thus the state must be pushed onto the stack before narrowing
# the clip path.  This is taken care of by GraphicsContextPdf.
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
# * Type 1 font support (i.e., "pdf.use_afm")
# * TTF support has lots of small TODOs, e.g. how do you know if a font
#   is serif/sans-serif, or symbolic/non-symbolic?
# * draw_markers, draw_line_collection, etc.
# * use_tex

def fill(strings, linelen=75):
    """Make one string from sequence of strings, with whitespace
    in between. The whitespace is chosen to form lines of at most
    linelen characters, if possible."""
    currpos = 0
    lasti = 0
    result = []
    for i, s in enumerate(strings):
        length = len(s)
        if currpos + length < linelen:
            currpos += length + 1
        else:
            result.append(' '.join(strings[lasti:i]))
            lasti = i
            currpos = length
    result.append(' '.join(strings[lasti:]))
    return '\n'.join(result)


def pdfRepr(obj):
    """Map Python objects to PDF syntax."""

    # Some objects defined later have their own pdfRepr method.
    if hasattr(obj, 'pdfRepr'):
        return obj.pdfRepr()

    # Floats. PDF does not have exponential notation (1.0e-10) so we
    # need to use %f with some precision.  Perhaps the precision
    # should adapt to the magnitude of the number?
    elif isinstance(obj, float):
        if not npy.isfinite(obj):
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
        r.extend(["%s %s" % (Name(key).pdfRepr(), pdfRepr(val))
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
        self.id = id            # object id
        self.len = len          # id of length object
        self.pdfFile = file
        self.file = file.fh     # file to which the stream is written
        self.compressobj = None # compression object
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

    def __init__(self, width, height, filename):
        self.width, self.height = width, height
        self.nextObject = 1     # next free object id
        self.xrefTable = [ [0, 65535, 'the zero object'] ]
        self.passed_in_file_object = False
        if is_string_like(filename):
            fh = file(filename, 'wb')
        elif is_writable_file_like(filename):
            fh = filename
            self.passed_in_file_object = True
        else:
            raise ValueError("filename must be a path or a file-like object")

        self.fh = fh
        self.currentstream = None # stream object to write to, if any
        fh.write("%PDF-1.4\n")    # 1.4 is the first version to have alpha
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
        self.nextFont = 1       # next free internal font name
        self.fontInfo = {}      # information on fonts: metrics, encoding

        self.alphaStates = {}   # maps alpha values to graphics state objects
        self.nextAlphaState = 1
        self.hatchPatterns = {}
        self.nextHatch = 1

        self.images = {}
        self.nextImage = 1

        self.markers = {}
        self.two_byte_charprocs = {}
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
        for name, value in self.markers.items():
            xobjects[name] = value[0]
        for name, value in self.two_byte_charprocs.items():
            xobjects[name] = value
        self.writeObject(self.XObjectObject, xobjects)
        self.writeImages()
        self.writeMarkers()
        self.writeXref()
        self.writeTrailer()
        if not self.passed_in_file_object:
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

    def fontName(self, fontprop):
        """
        Select a font based on fontprop and return a name suitable for
        Op.selectfont. If fontprop is a string, it will be interpreted
        as the filename of the font.
        """

        if is_string_like(fontprop):
            filename = fontprop
        elif rcParams['pdf.use14corefonts']:
            filename = findfont(fontprop, fontext='afm')
        else:
            filename = findfont(fontprop)

        Fx = self.fontNames.get(filename)
        if Fx is None:
            Fx = Name('F%d' % self.nextFont)
            self.fontNames[filename] = Fx
            self.nextFont += 1

        return Fx

    def writeFonts(self):
        fonts = {}
        for filename, Fx in self.fontNames.items():
            if filename.endswith('.afm'):
                fontdictObject = self._write_afm_font(filename)
            elif filename.endswith('.pfb') or filename.endswith('.pfa'):
                # a Type 1 font; limited support for now
                fontdictObject = self.embedType1(filename, self.fontInfo[Fx])
            else:
                realpath, stat_key = get_realpath_and_stat(filename)
                chars = self.used_characters.get(stat_key)
                if chars is not None and len(chars[1]):
                    fontdictObject = self.embedTTF(realpath, chars[1])
            fonts[Fx] = fontdictObject
            #print >>sys.stderr, filename
        self.writeObject(self.fontObject, fonts)

    def _write_afm_font(self, filename):
        fh = file(filename)
        font = AFM(fh)
        fh.close()
        fontname = font.get_fontname()
        fontdict = { 'Type': Name('Font'),
                     'Subtype': Name('Type1'),
                     'BaseFont': Name(fontname),
                     'Encoding': Name('WinAnsiEncoding') }
        fontdictObject = self.reserveObject('font dictionary')
        self.writeObject(fontdictObject, fontdict)
        return fontdictObject

    def embedType1(self, filename, fontinfo):
        # TODO: font effects such as SlantFont
        fh = open(filename, 'rb')
        matplotlib.verbose.report(
            'Embedding Type 1 font ' + filename, 'debug')
        try:
            fontdata = fh.read()
        finally:
            fh.close()

        font = FT2Font(filename)

        widthsObject, fontdescObject, fontdictObject, fontfileObject = \
            [ self.reserveObject(n) for n in
                ('font widths', 'font descriptor',
                 'font dictionary', 'font file') ]

        firstchar = 0
        lastchar = len(fontinfo.widths) - 1

        fontdict = {
            'Type':           Name('Font'),
            'Subtype':        Name('Type1'),
            'BaseFont':       Name(font.postscript_name),
            'FirstChar':      0,
            'LastChar':       lastchar,
            'Widths':         widthsObject,
            'FontDescriptor': fontdescObject,
            }

        if fontinfo.encodingfile is not None:
            enc = dviread.Encoding(fontinfo.encodingfile)
            differencesArray = [ Name(ch) for ch in enc ]
            differencesArray = [ 0 ] + differencesArray
            fontdict.update({
                    'Encoding': { 'Type': Name('Encoding'),
                                  'Differences': differencesArray },
                    })

        _, _, fullname, familyname, weight, italic_angle, fixed_pitch, \
            ul_position, ul_thickness = font.get_ps_font_info()

        flags = 0
        if fixed_pitch:   flags |= 1 << 0  # fixed width
        if 0:             flags |= 1 << 1  # TODO: serif
        if 1:             flags |= 1 << 2  # TODO: symbolic (most TeX fonts are)
        else:             flags |= 1 << 5  # non-symbolic
        if italic_angle:  flags |= 1 << 6  # italic
        if 0:             flags |= 1 << 16 # TODO: all caps
        if 0:             flags |= 1 << 17 # TODO: small caps
        if 0:             flags |= 1 << 18 # TODO: force bold

        descriptor = {
            'Type':        Name('FontDescriptor'),
            'FontName':    Name(font.postscript_name),
            'Flags':       flags,
            'FontBBox':    font.bbox,
            'ItalicAngle': italic_angle,
            'Ascent':      font.ascender,
            'Descent':     font.descender,
            'CapHeight':   1000, # TODO: find this out
            'XHeight':     500, # TODO: this one too
            'FontFile':    fontfileObject,
            'FontFamily':  familyname,
            'StemV':       50, # TODO
            # (see also revision 3874; but not all TeX distros have AFM files!)
            #'FontWeight': a number where 400 = Regular, 700 = Bold
            }

        self.writeObject(fontdictObject, fontdict)
        self.writeObject(widthsObject, fontinfo.widths)
        self.writeObject(fontdescObject, descriptor)

        t1font = type1font.Type1Font(filename)
        self.beginStream(fontfileObject.id, None,
                         { 'Length1': len(t1font.parts[0]),
                           'Length2': len(t1font.parts[1]),
                           'Length3': 0 })
        self.currentstream.write(t1font.parts[0])
        self.currentstream.write(t1font.parts[1])
        self.endStream()

        return fontdictObject

    def _get_xobject_symbol_name(self, filename, symbol_name):
        return "%s-%s" % (
            os.path.splitext(os.path.basename(filename))[0],
            symbol_name)

    _identityToUnicodeCMap = """/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
   /Ordering (UCS)
   /Supplement 0
>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <ffff>
endcodespacerange
%d beginbfrange
%s
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""

    def embedTTF(self, filename, characters):
        """Embed the TTF font from the named file into the document."""

        font = FT2Font(str(filename))
        fonttype = rcParams['pdf.fonttype']

        def cvt(length, upe=font.units_per_EM, nearest=True):
            "Convert font coordinates to PDF glyph coordinates"
            value = length / upe * 1000
            if nearest: return round(value)
            # Perhaps best to round away from zero for bounding
            # boxes and the like
            if value < 0: return floor(value)
            else: return ceil(value)

        def embedTTFType3(font, characters, descriptor):
            """The Type 3-specific part of embedding a Truetype font"""
            widthsObject = self.reserveObject('font widths')
            fontdescObject = self.reserveObject('font descriptor')
            fontdictObject = self.reserveObject('font dictionary')
            charprocsObject = self.reserveObject('character procs')
            differencesArray = []
            firstchar, lastchar = 0, 255
            bbox = [cvt(x, nearest=False) for x in font.bbox]

            fontdict = {
                'Type'            : Name('Font'),
                'BaseFont'        : ps_name,
                'FirstChar'       : firstchar,
                'LastChar'        : lastchar,
                'FontDescriptor'  : fontdescObject,
                'Subtype'         : Name('Type3'),
                'Name'            : descriptor['FontName'],
                'FontBBox'        : bbox,
                'FontMatrix'      : [ .001, 0, 0, .001, 0, 0 ],
                'CharProcs'       : charprocsObject,
                'Encoding'        : {
                    'Type'        : Name('Encoding'),
                    'Differences' : differencesArray},
                'Widths'          : widthsObject
                }

            # Make the "Widths" array
            from encodings import cp1252
            # The "decoding_map" was changed to a "decoding_table" as of Python 2.5.
            if hasattr(cp1252, 'decoding_map'):
                def decode_char(charcode):
                    return cp1252.decoding_map[charcode] or 0
            else:
                def decode_char(charcode):
                    return ord(cp1252.decoding_table[charcode])

            def get_char_width(charcode):
                unicode = decode_char(charcode)
                width = font.load_char(unicode, flags=LOAD_NO_SCALE|LOAD_NO_HINTING).horiAdvance
                return cvt(width)

            widths = [ get_char_width(charcode) for charcode in range(firstchar, lastchar+1) ]
            descriptor['MaxWidth'] = max(widths)

            # Make the "Differences" array, sort the ccodes < 255 from
            # the two-byte ccodes, and build the whole set of glyph ids
            # that we need from this font.
            cmap = font.get_charmap()
            glyph_ids = []
            differences = []
            two_byte_chars = Set()
            for c in characters:
                ccode = ord(c)
                gind = cmap.get(ccode) or 0
                glyph_ids.append(gind)
                glyph_name = font.get_glyph_name(gind)
                if ccode <= 255:
                    differences.append((ccode, glyph_name))
                else:
                    two_byte_chars.add(glyph_name)
            differences.sort()

            last_c = -2
            for c, name in differences:
                if c != last_c + 1:
                    differencesArray.append(c)
                differencesArray.append(Name(name))
                last_c = c

            # Make the charprocs array (using ttconv to generate the
            # actual outlines)
            rawcharprocs = ttconv.get_pdf_charprocs(filename, glyph_ids)
            charprocs = {}
            charprocsRef = {}
            for charname, stream in rawcharprocs.items():
                charprocDict = { 'Length': len(stream) }
                # The 2-byte characters are used as XObjects, so they
                # need extra info in their dictionary
                if charname in two_byte_chars:
                    charprocDict['Type'] = Name('XObject')
                    charprocDict['Subtype'] = Name('Form')
                    charprocDict['BBox'] = bbox
                charprocObject = self.reserveObject('charProc')
                self.beginStream(charprocObject.id, None, charprocDict)
                self.currentstream.write(stream)
                self.endStream()

                # Send the glyphs with ccode > 255 to the XObject dictionary,
                # and the others to the font itself
                if charname in two_byte_chars:
                    name = self._get_xobject_symbol_name(filename, charname)
                    self.two_byte_charprocs[name] = charprocObject
                else:
                    charprocs[charname] = charprocObject

            # Write everything out
            self.writeObject(fontdictObject, fontdict)
            self.writeObject(fontdescObject, descriptor)
            self.writeObject(widthsObject, widths)
            self.writeObject(charprocsObject, charprocs)

            return fontdictObject

        def embedTTFType42(font, characters, descriptor):
            """The Type 42-specific part of embedding a Truetype font"""
            fontdescObject = self.reserveObject('font descriptor')
            cidFontDictObject = self.reserveObject('CID font dictionary')
            type0FontDictObject = self.reserveObject('Type 0 font dictionary')
            cidToGidMapObject = self.reserveObject('CIDToGIDMap stream')
            fontfileObject = self.reserveObject('font file stream')
            wObject = self.reserveObject('Type 0 widths')
            toUnicodeMapObject = self.reserveObject('ToUnicode map')

            cidFontDict = {
                'Type'           : Name('Font'),
                'Subtype'        : Name('CIDFontType2'),
                'BaseFont'       : ps_name,
                'CIDSystemInfo'  : {
                    'Registry'   : 'Adobe',
                    'Ordering'   : 'Identity',
                    'Supplement' : 0 },
                'FontDescriptor' : fontdescObject,
                'W'              : wObject,
                'CIDToGIDMap'    : cidToGidMapObject
                }

            type0FontDict = {
                'Type'            : Name('Font'),
                'Subtype'         : Name('Type0'),
                'BaseFont'        : ps_name,
                'Encoding'        : Name('Identity-H'),
                'DescendantFonts' : [cidFontDictObject],
                'ToUnicode'       : toUnicodeMapObject
                }

            # Make fontfile stream
            descriptor['FontFile2'] = fontfileObject
            length1Object = self.reserveObject('decoded length of a font')
            self.beginStream(
                fontfileObject.id,
                self.reserveObject('length of font stream'),
                {'Length1': length1Object})
            fontfile = open(filename, 'rb')
            length1 = 0
            while True:
                data = fontfile.read(4096)
                if not data: break
                length1 += len(data)
                self.currentstream.write(data)
            fontfile.close()
            self.endStream()
            self.writeObject(length1Object, length1)

            # Make the 'W' (Widths) array, CidToGidMap and ToUnicode CMap
            # at the same time
            cid_to_gid_map = [u'\u0000'] * 65536
            cmap = font.get_charmap()
            unicode_mapping = []
            widths = []
            max_ccode = 0
            for c in characters:
                ccode = ord(c)
                gind = cmap.get(ccode) or 0
                glyph = font.load_char(ccode, flags=LOAD_NO_HINTING)
                # Why divided by 3.0 ??? Wish I knew... MGD
                widths.append((ccode, cvt(glyph.horiAdvance) / 3.0))
                if ccode < 65536:
                    cid_to_gid_map[ccode] = unichr(gind)
                max_ccode = max(ccode, max_ccode)
            widths.sort()
            cid_to_gid_map = cid_to_gid_map[:max_ccode + 1]

            last_ccode = -2
            w = []
            max_width = 0
            unicode_groups = []
            for ccode, width in widths:
                if ccode != last_ccode + 1:
                    w.append(ccode)
                    w.append([width])
                    unicode_groups.append([ccode, ccode])
                else:
                    w[-1].append(width)
                    unicode_groups[-1][1] = ccode
                max_width = max(max_width, width)
                last_ccode = ccode

            unicode_bfrange = []
            for start, end in unicode_groups:
                unicode_bfrange.append(
                    "<%04x> <%04x> [%s]" %
                    (start, end,
                     " ".join(["<%04x>" % x for x in range(start, end+1)])))
            unicode_cmap = (self._identityToUnicodeCMap %
                            (len(unicode_groups),
                             "\n".join(unicode_bfrange)))

            # CIDToGIDMap stream
            cid_to_gid_map = "".join(cid_to_gid_map).encode("utf-16be")
            self.beginStream(cidToGidMapObject.id,
                             None,
                             {'Length':  len(cid_to_gid_map)})
            self.currentstream.write(cid_to_gid_map)
            self.endStream()

            # ToUnicode CMap
            self.beginStream(toUnicodeMapObject.id,
                             None,
                             {'Length': unicode_cmap})
            self.currentstream.write(unicode_cmap)
            self.endStream()

            descriptor['MaxWidth'] = max_width

            # Write everything out
            self.writeObject(cidFontDictObject, cidFontDict)
            self.writeObject(type0FontDictObject, type0FontDict)
            self.writeObject(fontdescObject, descriptor)
            self.writeObject(wObject, w)

            return type0FontDictObject

        # Beginning of main embedTTF function...

        # You are lost in a maze of TrueType tables, all different...
        ps_name = Name(font.get_sfnt()[(1,0,0,6)])
        pclt = font.get_sfnt_table('pclt') \
            or { 'capHeight': 0, 'xHeight': 0 }
        post = font.get_sfnt_table('post') \
            or { 'italicAngle': (0,0) }
        ff = font.face_flags
        sf = font.style_flags

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
            'Type'        : Name('FontDescriptor'),
            'FontName'    : ps_name,
            'Flags'       : flags,
            'FontBBox'    : [ cvt(x, nearest=False) for x in font.bbox ],
            'Ascent'      : cvt(font.ascender, nearest=False),
            'Descent'     : cvt(font.descender, nearest=False),
            'CapHeight'   : cvt(pclt['capHeight'], nearest=False),
            'XHeight'     : cvt(pclt['xHeight']),
            'ItalicAngle' : post['italicAngle'][1], # ???
            'StemV'       : 0 # ???
            }

        # The font subsetting to a Type 3 font does not work for
        # OpenType (.otf) that embed a Postscript CFF font, so avoid that --
        # save as a (non-subsetted) Type 42 font instead.
        if is_opentype_cff_font(filename):
            fonttype = 42
            warnings.warn(("'%s' can not be subsetted into a Type 3 font. " +
                           "The entire font will be embedded in the output.") %
                           os.path.basename(filename))

        if fonttype == 3:
            return embedTTFType3(font, characters, descriptor)
        elif fonttype == 42:
            return embedTTFType42(font, characters, descriptor)

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
            if lst[2]:                # -
                for j in npy.arange(0.0, sidelen, density/lst[2]):
                    self.output(0, j, Op.moveto,
                                sidelen, j, Op.lineto)
            if lst[3]:                # /
                for j in npy.arange(0.0, sidelen, density/lst[3]):
                    self.output(0, j, Op.moveto,
                                sidelen-j, sidelen, Op.lineto,
                                sidelen-j, 0, Op.moveto,
                                sidelen, j, Op.lineto)
            if lst[4]:                # |
                for j in npy.arange(0.0, sidelen, density/lst[4]):
                    self.output(j, 0, Op.moveto,
                                j, sidelen, Op.lineto)
            if lst[5]:                # \
                for j in npy.arange(sidelen, 0.0, -density/lst[5]):
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

        rgba = npy.fromstring(s, npy.uint8)
        rgba.shape = (h, w, 4)
        rgb = rgba[:,:,:3]
        return h, w, rgb.tostring()

    def _gray(self, im, rc=0.3, gc=0.59, bc=0.11):
        rgbat = im.as_rgba_str()
        rgba = npy.fromstring(rgbat[2], npy.uint8)
        rgba.shape = (rgbat[0], rgbat[1], 4)
        rgba_f = rgba.astype(npy.float32)
        r = rgba_f[:,:,0]
        g = rgba_f[:,:,1]
        b = rgba_f[:,:,2]
        gray = (r*rc + g*gc + b*bc).astype(npy.uint8)
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

    #@staticmethod
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

    #@staticmethod
    def pathOperations(path):
        path.rewind(0)
        result = []
        while True:
            code, x, y = path.vertex()
            code = code & agg.path_cmd_mask
            if code == agg.path_cmd_stop:
                break
            elif code == agg.path_cmd_move_to:
                result += (x, y, Op.moveto)
            elif code == agg.path_cmd_line_to:
                result += (x, y, Op.lineto)
            elif code == agg.path_cmd_curve3:
                pass # TODO
            elif code == agg.path_cmd_curve4:
                pass # TODO
            elif code == agg.path_cmd_end_poly:
                result += (Op.closepath,)
            else:
                print >>sys.stderr, "pathOperations", code, xp, yp
        return result
    pathOperations = staticmethod(pathOperations)

    def writePath(self, path, fillp):
        self.output(*self.pathOperations(path))
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
    truetype_font_cache = maxdict(50)
    afm_font_cache = maxdict(50)

    def __init__(self, file, dpi):
        RendererBase.__init__(self)
        self.file = file
        self.gc = self.new_gc()
        self.file.used_characters = self.used_characters = {}
        self.mathtext_parser = MathTextParser("Pdf")
        self.image_magnification = dpi/72.0
        self.tex_font_map = None

    def finalize(self):
        self.gc.finalize()

    def check_gc(self, gc, fillcolor=None):
        orig_fill = gc._fillcolor
        gc._fillcolor = fillcolor

        delta = self.gc.delta(gc)
        if delta: self.file.output(*delta)

        # Restore gc to avoid unwanted side effects
        gc._fillcolor = orig_fill

    def tex_font_mapping(self, texfont):
        if self.tex_font_map is None:
            self.tex_font_map = \
                dviread.PsfontsMap(dviread.find_tex_file('pdftex.map'))
        return self.tex_font_map[texfont]

    def track_characters(self, font, s):
        """Keeps track of which characters are required from
        each font."""
        if isinstance(font, (str, unicode)):
            fname = font
        else:
            fname = font.fname
        realpath, stat_key = get_realpath_and_stat(fname)
        used_characters = self.used_characters.setdefault(
            stat_key, (realpath, Set()))
        used_characters[1].update(s)

    def merge_used_characters(self, other):
        for stat_key, (realpath, set) in other.items():
            used_characters = self.used_characters.setdefault(
                stat_key, (realpath, Set()))
            used_characters[1].update(set)

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

        def arc_to_bezier(cx, cy, rx, ry, angle1, sweep, rotation):
            halfsweep = sweep / 2.0
            x0, y0 = cos(halfsweep), sin(halfsweep)
            tx = (1.0 - x0) * 4.0/3.0;
            ty = y0 - tx * x0 / y0;
            px =  x0, x0+tx, x0+tx, x0
            py = -y0,   -ty,    ty, y0
            sn, cs = sin(angle1 + halfsweep), cos(angle1 + halfsweep)
            result = [ (rx * (pxi * cs - pyi * sn),
                        ry * (pxi * sn + pyi * cs))
                       for pxi, pyi in zip(px, py) ]
            result = [ (cx + cos(rotation)*x - sin(rotation)*y,
                        cy + sin(rotation)*x + cos(rotation)*y)
                       for x, y in result ]
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
        bp = [ pi/2.0 * i
               for i in range(4)
               if pi/2.0 * i < sweep-epsilon ]
        bp.append(sweep)
        subarcs = [ arc_to_bezier(x, y, width/2.0, height/2.0,
                                  bp[i], bp[i+1]-bp[i], rotation)
                    for i in range(len(bp)-1) ]

        self.check_gc(gcEdge, rgbFace)
        self.file.output(subarcs[0][0], subarcs[0][1], Op.moveto)
        for arc in subarcs:
            self.file.output(*(arc[2:] + (Op.curveto,)))

        self.file.output(self.gc.close_and_paint())

    def draw_path(self, gc, rgbFace, path):
        self.check_gc(gc, rgbFace)

        cmds = []

        while 1:
            code, xp, yp = path.vertex()

            if code == agg.path_cmd_stop:
                cmds.append(Op.closepath)
                break
            elif code == agg.path_cmd_move_to:
                cmds.extend([xp, yp, Op.moveto])
            elif code == agg.path_cmd_line_to:
                cmds.extend([xp, yp, Op.lineto])
            elif code == agg.path_cmd_curve3:
                cmds.extend([xp, yp])
                cmds.extend([xp, yp])
                cmds.extend(path.vertex()[1:])
                cmds.append(Op.curveto)
            elif code == agg.path_cmd_curve4:
                cmds.extend([xp, yp])
                cmds.extend(path.vertex()[1:])
                cmds.extend(path.vertex()[1:])
                cmds.append(Op.curveto)
            elif code == agg.path_cmd_end_poly:
                cmds.append(Op.closepath)
        self.file.output(*cmds)
        self.file.output(self.gc.paint())

    def get_image_magnification(self):
        return self.image_magnification

    def draw_image(self, x, y, im, bbox):
        #print >>sys.stderr, "draw_image called"

        gc = self.new_gc()
        gc.set_clip_rectangle(bbox.get_bounds())
        self.check_gc(gc)

        h, w = im.get_size_out()
        h, w = h/self.image_magnification, w/self.image_magnification
        imob = self.file.imageObject(im)
        self.file.output(Op.gsave, w, 0, 0, h, x, y, Op.concat_matrix,
                         imob, Op.use_xobject, Op.grestore)

    def draw_line(self, gc, x1, y1, x2, y2):
        if npy.isnan(x1) or npy.isnan(x2) or npy.isnan(y1) or npy.isnan(y2):
            return
        self.check_gc(gc)
        self.file.output(x1, y1, Op.moveto,
                         x2, y2, Op.lineto, self.gc.paint())

    def draw_lines(self, gc, x, y, transform=None):
        self.check_gc(gc)
        if transform is not None:
            x, y = transform.seq_x_y(x, y)
        nan_at = npy.isnan(x) | npy.isnan(y)
        next_op = Op.moveto
        for i in range(len(x)):
            if nan_at[i]:
                next_op = Op.moveto
            else:
                self.file.output(x[i], y[i], next_op)
                next_op = Op.lineto
        self.file.output(self.gc.paint())

    def draw_point(self, gc, x, y):
        print >>sys.stderr, "draw_point called"

        self.check_gc(gc, gc._rgb)
        self.file.output(x, y, 1, 1,
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
        self.file.output(points[0][0], points[0][1], Op.moveto)
        for x,y in points[1:]:
            self.file.output(x, y, Op.lineto)
        self.file.output(self.gc.close_and_paint())

    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
        self.check_gc(gcEdge, rgbFace)
        self.file.output(x, y, width, height, Op.rectangle)
        self.file.output(self.gc.paint())

    def draw_markers(self, gc, path, rgbFace, x, y, trans):
        self.check_gc(gc, rgbFace)
        fillp = rgbFace is not None
        marker = self.file.markerObject(path, fillp, self.gc._linewidth)
        x, y = trans.numerix_x_y(npy.asarray(x), npy.asarray(y))
        nan_at = npy.isnan(x) | npy.isnan(y)

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
        if angle == oldangle == 0:
            self.file.output(x - oldx, y - oldy, Op.textpos)
        else:
            angle = angle / 180.0 * pi
            self.file.output( cos(angle), sin(angle),
                             -sin(angle), cos(angle),
                              x,        y,         Op.textmatrix)

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        # TODO: fix positioning and encoding
        width, height, descent, glyphs, rects, used_characters = \
            self.mathtext_parser.parse(s, 72, prop)
        self.merge_used_characters(used_characters)

        # When using Type 3 fonts, we can't use character codes higher
        # than 255, so we use the "Do" command to render those
        # instead.
        global_fonttype = rcParams['pdf.fonttype']

        # Set up a global transformation matrix for the whole math expression
        a = angle / 180.0 * pi
        self.file.output(Op.gsave)
        self.file.output(cos(a), sin(a), -sin(a), cos(a), x, y,
                         Op.concat_matrix)

        self.check_gc(gc, gc._rgb)
        self.file.output(Op.begin_text)
        prev_font = None, None
        oldx, oldy = 0, 0
        for ox, oy, fontname, fontsize, num, symbol_name in glyphs:
            if is_opentype_cff_font(fontname):
                fonttype = 42
            else:
                fonttype = global_fonttype

            if fonttype == 42 or num <= 255:
                self._setup_textpos(ox, oy, 0, oldx, oldy)
                oldx, oldy = ox, oy
                if (fontname, fontsize) != prev_font:
                    self.file.output(self.file.fontName(fontname), fontsize,
                                     Op.selectfont)
                    prev_font = fontname, fontsize
                self.file.output(self.encode_string(unichr(num), fonttype), Op.show)
        self.file.output(Op.end_text)

        # If using Type 3 fonts, render all of the two-byte characters
        # as XObjects using the 'Do' command.
        if global_fonttype == 3:
            for ox, oy, fontname, fontsize, num, symbol_name in glyphs:
                if is_opentype_cff_font(fontname):
                    fonttype = 42
                else:
                    fonttype = global_fonttype

                if fonttype == 3 and num > 255:
                    self.file.fontName(fontname)
                    self.file.output(Op.gsave,
                                     0.001 * fontsize, 0,
                                     0, 0.001 * fontsize,
                                     ox, oy, Op.concat_matrix)
                    name = self.file._get_xobject_symbol_name(
                        fontname, symbol_name)
                    self.file.output(Name(name), Op.use_xobject)
                    self.file.output(Op.grestore)

        # Draw any horizontal lines in the math layout
        for ox, oy, width, height in rects:
            self.file.output(Op.gsave, ox, oy, width, height,
                             Op.rectangle, Op.fill, Op.grestore)

        # Pop off the global transformation
        self.file.output(Op.grestore)

    def draw_tex(self, gc, x, y, s, prop, angle):
        texmanager = self.get_texmanager()
        fontsize = prop.get_size_in_points()
        dvifile = texmanager.make_dvi(s, fontsize)
        dvi = dviread.Dvi(dvifile, 72)
        page = iter(dvi).next()
        dvi.close()

        if angle == 0:          # avoid rounding errors in common case
            def mytrans(x1, y1):
                return x+x1, y+y1
        else:
            def mytrans(x1, y1, x=x, y=y, a=angle / 180.0 * pi):
                return x + cos(a)*x1 - sin(a)*y1, \
                       y + sin(a)*x1 + cos(a)*y1

        # Gather font information and do some setup for combining
        # characters into strings.
        oldfont, seq = None, []
        for x1, y1, dvifont, glyph, width in page.text:
            if dvifont != oldfont:
                psfont = self.tex_font_mapping(dvifont.texname)
                pdfname = self.file.fontName(psfont.filename)
                if self.file.fontInfo.get(pdfname, None) is None:
                    self.file.fontInfo[pdfname] = Bunch(
                        encodingfile=psfont.encoding,
                        widths=dvifont.widths,
                        dvifont=dvifont)
                seq += [['font', pdfname, dvifont.size]]
                oldfont = dvifont
            seq += [['text', x1, y1, [chr(glyph)], x1+width]]
        seq += [('end',)]

        # Find consecutive text strings with constant x coordinate and
        # combine into a sequence of strings and kerns, or just one
        # string (if any kerns would be less than 0.1 points).
        i, curx = 0, 0
        while i < len(seq)-1:
            elt, next = seq[i:i+2]
            if elt[0] == next[0] == 'text' and elt[2] == next[2]:
                offset = elt[4] - next[1]
                if abs(offset) < 0.1:
                    elt[3][-1] += next[3][0]
                    elt[4] += next[4]-next[1]
                else:
                    elt[3] += [offset*1000.0/dvifont.size, next[3][0]]
                    elt[4] = next[4]
                del seq[i+1]
                continue
            i += 1

        # Now do the actual output.
        self.check_gc(gc, gc._rgb)
        self.file.output(Op.begin_text)
        curx, cury, oldx, oldy = 0, 0, 0, 0
        for elt in seq:
            if elt[0] == 'font':
                self.file.output(elt[1], elt[2], Op.selectfont)
            elif elt[0] == 'text':
                curx, cury = mytrans(elt[1], elt[2])
                self._setup_textpos(curx, cury, angle, oldx, oldy)
                oldx, oldy = curx, cury
                if len(elt[3]) == 1:
                    self.file.output(elt[3][0], Op.show)
                else:
                    self.file.output(elt[3], Op.showkern)
            else:
                assert elt[0] == 'end'
        self.file.output(Op.end_text)

        # Finally output the boxes (used for the variable-length lines
        # in square roots and the like).
        boxgc = self.new_gc()
        boxgc.copy_properties(gc)
        boxgc.set_linewidth(0)
        for x1, y1, h, w in page.boxes:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = \
                mytrans(x1, y1), mytrans(x1+w, y1), \
                mytrans(x1+w, y1+h), mytrans(x1, y1+h)
            self.draw_polygon(boxgc, gc._rgb,
                              ((x1,y1), (x2,y2), (x3,y3), (x4,y4)))

    def encode_string(self, s, fonttype):
        if fonttype == 3:
            return s.encode('cp1252', 'replace')
        return s.encode('utf-16be', 'replace')

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
        # TODO: combine consecutive texts into one BT/ET delimited section

        # This function is rather complex, since there is no way to
        # access characters of a Type 3 font with codes > 255.  (Type
        # 3 fonts can not have a CIDMap).  Therefore, we break the
        # string into chunks, where each chunk contains exclusively
        # 1-byte or exclusively 2-byte characters, and output each
        # chunk a separate command.  1-byte characters use the regular
        # text show command (Tj), whereas 2-byte characters use the
        # use XObject command (Do).  If using Type 42 fonts, all of
        # this complication is avoided, but of course, those fonts can
        # not be subsetted.

        self.check_gc(gc, gc._rgb)
        if ismath: return self.draw_mathtext(gc, x, y, s, prop, angle)

        fontsize = prop.get_size_in_points()

        if rcParams['pdf.use14corefonts']:
            font = self._get_font_afm(prop)
            l, b, w, h = font.get_str_bbox(s)
            y -= b * fontsize / 1000
            fonttype = 42
        else:
            font = self._get_font_ttf(prop)
            self.track_characters(font, s)
            font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
            y += font.get_descent() / 64.0

            fonttype = rcParams['pdf.fonttype']

            # We can't subset all OpenType fonts, so switch to Type 42
            # in that case.
            if is_opentype_cff_font(font.fname):
                fonttype = 42

        def check_simple_method(s):
            """Determine if we should use the simple or woven method
            to output this text, and chunks the string into 1-byte and
            2-byte sections if necessary."""
            use_simple_method = True
            chunks = []

            if not rcParams['pdf.use14corefonts']:
                if fonttype == 3 and not isinstance(s, str) and len(s) != 0:
                    # Break the string into chunks where each chunk is either
                    # a string of chars <= 255, or a single character > 255.
                    s = unicode(s)
                    for c in s:
                        if ord(c) <= 255:
                            char_type = 1
                        else:
                            char_type = 2
                        if len(chunks) and chunks[-1][0] == char_type:
                            chunks[-1][1].append(c)
                        else:
                            chunks.append((char_type, [c]))
                    use_simple_method = (len(chunks) == 1
                                         and chunks[-1][0] == 1)
            return use_simple_method, chunks

        def draw_text_simple():
            """Outputs text using the simple method."""
            self.file.output(Op.begin_text,
                             self.file.fontName(prop),
                             prop.get_size_in_points(),
                             Op.selectfont)
            self._setup_textpos(x, y, angle)
            self.file.output(self.encode_string(s, fonttype), Op.show, Op.end_text)

        def draw_text_woven(chunks):
            """Outputs text using the woven method, alternating
            between chunks of 1-byte characters and 2-byte characters.
            Only used for Type 3 fonts."""
            chunks = [(a, ''.join(b)) for a, b in chunks]
            cmap = font.get_charmap()

            # Do the rotation and global translation as a single matrix
            # concatenation up front
            self.file.output(Op.gsave)
            a = angle / 180.0 * pi
            self.file.output(cos(a), sin(a), -sin(a), cos(a), x, y,
                             Op.concat_matrix)

            # Output all the 1-byte characters in a BT/ET group, then
            # output all the 2-byte characters.
            for mode in (1, 2):
                newx = oldx = 0
                # Output a 1-byte character chunk
                if mode == 1:
                    self.file.output(Op.begin_text,
                                     self.file.fontName(prop),
                                     prop.get_size_in_points(),
                                     Op.selectfont)

                for chunk_type, chunk in chunks:
                    if mode == 1 and chunk_type == 1:
                        self._setup_textpos(newx, 0, 0, oldx, 0, 0)
                        self.file.output(self.encode_string(chunk, fonttype), Op.show)
                        oldx = newx

                    lastgind = None
                    for c in chunk:
                        ccode = ord(c)
                        gind = cmap.get(ccode)
                        if gind is not None:
                            if mode == 2 and chunk_type == 2:
                                glyph_name = font.get_glyph_name(gind)
                                self.file.output(Op.gsave)
                                self.file.output(0.001 * fontsize, 0,
                                                 0, 0.001 * fontsize,
                                                 newx, 0, Op.concat_matrix)
                                name = self.file._get_xobject_symbol_name(
                                    font.fname, glyph_name)
                                self.file.output(Name(name), Op.use_xobject)
                                self.file.output(Op.grestore)

                            # Move the pointer based on the character width
                            # and kerning
                            glyph = font.load_char(ccode, flags=LOAD_NO_HINTING)
                            if lastgind is not None:
                                kern = font.get_kerning(
                                    lastgind, gind, KERNING_UNFITTED)
                            else:
                                kern = 0
                            lastgind = gind
                            newx += kern/64.0 + glyph.linearHoriAdvance/65536.0

                if mode == 1:
                    self.file.output(Op.end_text)

            self.file.output(Op.grestore)

        use_simple_method, chunks = check_simple_method(s)
        if use_simple_method:
            return draw_text_simple()
        else:
            return draw_text_woven(chunks)

    def get_text_width_height_descent(self, s, prop, ismath):
        if rcParams['text.usetex']:
            texmanager = self.get_texmanager()
            fontsize = prop.get_size_in_points()
            dvifile = texmanager.make_dvi(s, fontsize)
            dvi = dviread.Dvi(dvifile, 72)
            page = iter(dvi).next()
            dvi.close()
            return page.width, page.height, page.descent
        if ismath:
            w, h, d, glyphs, rects, used_characters = \
                self.mathtext_parser.parse(s, 72, prop)

        elif rcParams['pdf.use14corefonts']:
            font = self._get_font_afm(prop)
            l, b, w, h, d = font.get_str_bbox_and_descent(s)
            scale = prop.get_size_in_points() / 1000.0
            w *= scale
            h *= scale
            d *= scale
        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0.0, flags=LOAD_NO_HINTING)
            w, h = font.get_width_height()
            w /= 64.0
            h /= 64.0
            d = font.get_descent()
            d /= 64.0
        return w, h, d

    def _get_font_afm(self, prop):
        key = hash(prop)
        font = self.afm_font_cache.get(key)
        if font is None:
            filename = findfont(prop, fontext='afm')
            font = self.afm_font_cache.get(filename)
            if font is None:
                fh = file(filename)
                font = AFM(fh)
                self.afm_font_cache[filename] = font
                fh.close()
            self.afm_font_cache[key] = font
        return font

    def _get_font_ttf(self, prop):
        key = hash(prop)
        font = self.truetype_font_cache.get(key)
        if font is None:
            filename = findfont(prop)
            font = self.truetype_font_cache.get(filename)
            if font is None:
                font = FT2Font(str(filename))
                self.truetype_font_cache[filename] = font
            self.truetype_font_cache[key] = font
        font.clear()
        font.set_size(prop.get_size_in_points(), 72.0)
        return font

    def flipy(self):
        return False

    def get_canvas_width_height(self):
        return self.file.width / 72.0, self.file.height / 72.0

    def new_gc(self):
        return GraphicsContextPdf(self.file)


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
        if rcParams['pdf.inheritcolor']:
            return []
        if rgb[0] == rgb[1] == rgb[2]:
            return [rgb[0], Op.setgray_stroke]
        else:
            return list(rgb) + [Op.setrgb_stroke]

    def fillcolor_cmd(self, rgb):
        if rgb is None or rcParams['pdf.inheritcolor']:
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

    def clip_cmd(self, cliprect, clippath):
        """Set clip rectangle. Calls self.pop() and self.push()."""
        cmds = []
        # Pop graphics state until we hit the right one or the stack is empty
        while (self._cliprect, self._clippath) != (cliprect, clippath) \
                and self.parent is not None:
            cmds.extend(self.pop())
        # Unless we hit the right one, set the clip polygon
        if (self._cliprect, self._clippath) != (cliprect, clippath):
            cmds.extend(self.push())
            if self._cliprect != cliprect:
                cmds.extend([t for t in cliprect] +
                            [Op.rectangle, Op.clip, Op.endpath])
            if self._clippath != clippath:
                cmds.extend(PdfFile.pathOperations(clippath) +
                            [Op.clip, Op.endpath])
        return cmds

    commands = (
        (('_cliprect', '_clippath'), clip_cmd), # must come first since may pop
        (('_alpha',), alpha_cmd),
        (('_capstyle',), capstyle_cmd),
        (('_fillcolor',), fillcolor_cmd),
        (('_joinstyle',), joinstyle_cmd),
        (('_linewidth',), linewidth_cmd),
        (('_dashes',), dash_cmd),
        (('_rgb',), rgb_cmd),
        (('_hatch',), hatch_cmd),  # must come after fillcolor and rgb
        )

    # TODO: _linestyle

    def delta(self, other):
        """
        Copy properties of other into self and return PDF commands
        needed to transform self into other.
        """
        cmds = []
        for params, cmd in self.commands:
            ours = [ getattr(self, p) for p in params ]
            theirs = [ getattr(other, p) for p in params ]
            if ours != theirs:
                cmds.extend(cmd(self, *theirs))
                for p in params:
                    setattr(self, p, getattr(other, p))
        return cmds

    def copy_properties(self, other):
        """
        Copy properties of other into self.
        """
        GraphicsContextBase.copy_properties(self, other)
        self._fillcolor = other._fillcolor

    def finalize(self):
        """
        Make sure every pushed graphics state is popped.
        """
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

    def draw(self):
        pass

    filetypes = {'pdf': 'Portable Document Format'}

    def get_default_filetype(self):
        return 'pdf'

    def print_pdf(self, filename, **kwargs):
        dpi = kwargs.get('dpi', None)
        self.figure.set_dpi(72) # Override the dpi kwarg
        width, height = self.figure.get_size_inches()
        file = PdfFile(width, height, filename)
        renderer = RendererPdf(file, dpi)
        self.figure.draw(renderer)
        renderer.finalize()
        file.close()

class FigureManagerPdf(FigureManagerBase):
    pass

FigureManager = FigureManagerPdf
