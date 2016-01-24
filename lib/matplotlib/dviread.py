"""
An experimental module for reading dvi files output by TeX. Several
limitations make this not (currently) useful as a general-purpose dvi
preprocessor, but it is currently used by the pdf backend for
processing usetex text.

Interface::

  dvi = Dvi(filename, 72)
  # iterate over pages (but only one page is supported for now):
  for page in dvi:
      w, h, d = page.width, page.height, page.descent
      for x,y,font,glyph,width in page.text:
          fontname = font.texname
          pointsize = font.size
          ...
      for x,y,height,width in page.boxes:
          ...

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import xrange

import errno
import matplotlib
import matplotlib.cbook as mpl_cbook
from matplotlib.compat import subprocess
from matplotlib import rcParams
import numpy as np
import struct
import sys
import os

if six.PY3:
    def ord(x):
        return x

_dvistate = mpl_cbook.Bunch(pre=0, outer=1, inpage=2, post_post=3, finale=4)

class Dvi(object):
    """
    A dvi ("device-independent") file, as produced by TeX.
    The current implementation only reads the first page and does not
    even attempt to verify the postamble.
    """

    def __init__(self, filename, dpi):
        """
        Initialize the object. This takes the filename as input and
        opens the file; actually reading the file happens when
        iterating through the pages of the file.
        """
        matplotlib.verbose.report('Dvi: ' + filename, 'debug')
        self.file = open(filename, 'rb')
        self.dpi = dpi
        self.fonts = {}
        self.state = _dvistate.pre
        self.baseline = self._get_baseline(filename)

    def _get_baseline(self, filename):
        if rcParams['text.latex.preview']:
            base, ext = os.path.splitext(filename)
            baseline_filename = base + ".baseline"
            if os.path.exists(baseline_filename):
                with open(baseline_filename, 'rb') as fd:
                    l = fd.read().split()
                height, depth, width = l
                return float(depth)
        return None

    def __iter__(self):
        """
        Iterate through the pages of the file.

        Returns (text, boxes) pairs, where:
          text is a list of (x, y, fontnum, glyphnum, width) tuples
          boxes is a list of (x, y, height, width) tuples

        The coordinates are transformed into a standard Cartesian
        coordinate system at the dpi value given when initializing.
        The coordinates are floating point numbers, but otherwise
        precision is not lost and coordinate values are not clipped to
        integers.
        """
        while True:
            have_page = self._read()
            if have_page:
                yield self._output()
            else:
                break

    def close(self):
        """
        Close the underlying file if it is open.
        """
        if not self.file.closed:
            self.file.close()

    def _output(self):
        """
        Output the text and boxes belonging to the most recent page.
        page = dvi._output()
        """
        minx, miny, maxx, maxy = np.inf, np.inf, -np.inf, -np.inf
        maxy_pure = -np.inf
        for elt in self.text + self.boxes:
            if len(elt) == 4:   # box
                x,y,h,w = elt
                e = 0           # zero depth
            else:               # glyph
                x,y,font,g,w = elt
                h,e = font._height_depth_of(g)
            minx = min(minx, x)
            miny = min(miny, y - h)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + e)
            maxy_pure = max(maxy_pure, y)

        if self.dpi is None:
            # special case for ease of debugging: output raw dvi coordinates
            return mpl_cbook.Bunch(text=self.text, boxes=self.boxes,
                                   width=maxx-minx, height=maxy_pure-miny,
                                   descent=descent)

        d = self.dpi / (72.27 * 2**16) # from TeX's "scaled points" to dpi units
        if self.baseline is None:
            descent = (maxy - maxy_pure) * d
        else:
            descent = self.baseline

        text =  [ ((x-minx)*d, (maxy-y)*d - descent, f, g, w*d)
                  for (x,y,f,g,w) in self.text ]
        boxes = [ ((x-minx)*d, (maxy-y)*d - descent, h*d, w*d) for (x,y,h,w) in self.boxes ]

        return mpl_cbook.Bunch(text=text, boxes=boxes,
                               width=(maxx-minx)*d,
                               height=(maxy_pure-miny)*d,
                               descent=descent)

    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        while True:
            byte = ord(self.file.read(1)[0])
            self._dispatch(byte)
            if byte == 140: # end of page
                return True
            if self.state == _dvistate.post_post: # end of file
                self.close()
                return False

    def _arg(self, nbytes, signed=False):
        """
        Read and return an integer argument *nbytes* long.
        Signedness is determined by the *signed* keyword.
        """
        str = self.file.read(nbytes)
        value = ord(str[0])
        if signed and value >= 0x80:
            value = value - 0x100
        for i in range(1, nbytes):
            value = 0x100*value + ord(str[i])
        return value

    def _dispatch(self, byte):
        """
        Based on the opcode *byte*, read the correct kinds of
        arguments from the dvi file and call the method implementing
        that opcode with those arguments.
        """
        if 0 <= byte <= 127: self._set_char(byte)
        elif byte == 128: self._set_char(self._arg(1))
        elif byte == 129: self._set_char(self._arg(2))
        elif byte == 130: self._set_char(self._arg(3))
        elif byte == 131: self._set_char(self._arg(4, True))
        elif byte == 132: self._set_rule(self._arg(4, True), self._arg(4, True))
        elif byte == 133: self._put_char(self._arg(1))
        elif byte == 134: self._put_char(self._arg(2))
        elif byte == 135: self._put_char(self._arg(3))
        elif byte == 136: self._put_char(self._arg(4, True))
        elif byte == 137: self._put_rule(self._arg(4, True), self._arg(4, True))
        elif byte == 138: self._nop()
        elif byte == 139: self._bop(*[self._arg(4, True) for i in range(11)])
        elif byte == 140: self._eop()
        elif byte == 141: self._push()
        elif byte == 142: self._pop()
        elif byte == 143: self._right(self._arg(1, True))
        elif byte == 144: self._right(self._arg(2, True))
        elif byte == 145: self._right(self._arg(3, True))
        elif byte == 146: self._right(self._arg(4, True))
        elif byte == 147: self._right_w(None)
        elif byte == 148: self._right_w(self._arg(1, True))
        elif byte == 149: self._right_w(self._arg(2, True))
        elif byte == 150: self._right_w(self._arg(3, True))
        elif byte == 151: self._right_w(self._arg(4, True))
        elif byte == 152: self._right_x(None)
        elif byte == 153: self._right_x(self._arg(1, True))
        elif byte == 154: self._right_x(self._arg(2, True))
        elif byte == 155: self._right_x(self._arg(3, True))
        elif byte == 156: self._right_x(self._arg(4, True))
        elif byte == 157: self._down(self._arg(1, True))
        elif byte == 158: self._down(self._arg(2, True))
        elif byte == 159: self._down(self._arg(3, True))
        elif byte == 160: self._down(self._arg(4, True))
        elif byte == 161: self._down_y(None)
        elif byte == 162: self._down_y(self._arg(1, True))
        elif byte == 163: self._down_y(self._arg(2, True))
        elif byte == 164: self._down_y(self._arg(3, True))
        elif byte == 165: self._down_y(self._arg(4, True))
        elif byte == 166: self._down_z(None)
        elif byte == 167: self._down_z(self._arg(1, True))
        elif byte == 168: self._down_z(self._arg(2, True))
        elif byte == 169: self._down_z(self._arg(3, True))
        elif byte == 170: self._down_z(self._arg(4, True))
        elif 171 <= byte <= 234: self._fnt_num(byte-171)
        elif byte == 235: self._fnt_num(self._arg(1))
        elif byte == 236: self._fnt_num(self._arg(2))
        elif byte == 237: self._fnt_num(self._arg(3))
        elif byte == 238: self._fnt_num(self._arg(4, True))
        elif 239 <= byte <= 242:
            len = self._arg(byte-238)
            special = self.file.read(len)
            self._xxx(special)
        elif 243 <= byte <= 246:
            k = self._arg(byte-242, byte==246)
            c, s, d, a, l = [ self._arg(x) for x in (4, 4, 4, 1, 1) ]
            n = self.file.read(a+l)
            self._fnt_def(k, c, s, d, a, l, n)
        elif byte == 247:
            i, num, den, mag, k = [ self._arg(x) for x in (1, 4, 4, 4, 1) ]
            x = self.file.read(k)
            self._pre(i, num, den, mag, x)
        elif byte == 248: self._post()
        elif byte == 249: self._post_post()
        else:
            raise ValueError("unknown command: byte %d"%byte)

    def _pre(self, i, num, den, mag, comment):
        if self.state != _dvistate.pre:
            raise ValueError("pre command in middle of dvi file")
        if i != 2:
            raise ValueError("Unknown dvi format %d"%i)
        if num != 25400000 or den != 7227 * 2**16:
            raise ValueError("nonstandard units in dvi file")
            # meaning: TeX always uses those exact values, so it
            # should be enough for us to support those
            # (There are 72.27 pt to an inch so 7227 pt =
            # 7227 * 2**16 sp to 100 in. The numerator is multiplied
            # by 10^5 to get units of 10**-7 meters.)
        if mag != 1000:
            raise ValueError("nonstandard magnification in dvi file")
            # meaning: LaTeX seems to frown on setting \mag, so
            # I think we can assume this is constant
        self.state = _dvistate.outer

    def _set_char(self, char):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced set_char in dvi file")
        self._put_char(char)
        self.h += self.fonts[self.f]._width_of(char)

    def _set_rule(self, a, b):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced set_rule in dvi file")
        self._put_rule(a, b)
        self.h += b

    def _put_char(self, char):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced put_char in dvi file")
        font = self.fonts[self.f]
        if font._vf is None:
            self.text.append((self.h, self.v, font, char,
                              font._width_of(char)))
        else:
            scale = font._scale
            for x, y, f, g, w in font._vf[char].text:
                newf = DviFont(scale=_mul2012(scale, f._scale),
                               tfm=f._tfm, texname=f.texname, vf=f._vf)
                self.text.append((self.h + _mul2012(x, scale),
                                  self.v + _mul2012(y, scale),
                                  newf, g, newf._width_of(g)))
            self.boxes.extend([(self.h + _mul2012(x, scale),
                                self.v + _mul2012(y, scale),
                                _mul2012(a, scale), _mul2012(b, scale))
                               for x, y, a, b in font._vf[char].boxes])

    def _put_rule(self, a, b):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced put_rule in dvi file")
        if a > 0 and b > 0:
            self.boxes.append((self.h, self.v, a, b))

    def _nop(self):
        pass

    def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
        if self.state != _dvistate.outer:
            raise ValueError("misplaced bop in dvi file (state %d)" % self.state)
        self.state = _dvistate.inpage
        self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
        self.stack = []
        self.text = []          # list of (x,y,fontnum,glyphnum)
        self.boxes = []         # list of (x,y,width,height)

    def _eop(self):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced eop in dvi file")
        self.state = _dvistate.outer
        del self.h, self.v, self.w, self.x, self.y, self.z, self.stack

    def _push(self):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced push in dvi file")
        self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))

    def _pop(self):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced pop in dvi file")
        self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()

    def _right(self, b):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced right in dvi file")
        self.h += b

    def _right_w(self, new_w):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced w in dvi file")
        if new_w is not None:
            self.w = new_w
        self.h += self.w

    def _right_x(self, new_x):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced x in dvi file")
        if new_x is not None:
            self.x = new_x
        self.h += self.x

    def _down(self, a):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced down in dvi file")
        self.v += a

    def _down_y(self, new_y):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced y in dvi file")
        if new_y is not None:
            self.y = new_y
        self.v += self.y

    def _down_z(self, new_z):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced z in dvi file")
        if new_z is not None:
            self.z = new_z
        self.v += self.z

    def _fnt_num(self, k):
        if self.state != _dvistate.inpage:
            raise ValueError("misplaced fnt_num in dvi file")
        self.f = k

    def _xxx(self, special):
        if six.PY3:
            matplotlib.verbose.report(
                'Dvi._xxx: encountered special: %s'
                % ''.join([(32 <= ord(ch) < 127) and chr(ch)
                           or '<%02x>' % ord(ch)
                           for ch in special]),
                'debug')
        else:
            matplotlib.verbose.report(
                'Dvi._xxx: encountered special: %s'
                % ''.join([(32 <= ord(ch) < 127) and ch
                           or '<%02x>' % ord(ch)
                           for ch in special]),
                'debug')

    def _fnt_def(self, k, c, s, d, a, l, n):
        fontname = n[-l:].decode('ascii')
        tfm = _tfmfile(fontname)
        if tfm is None:
            if six.PY2:
                error_class = OSError
            else:
                error_class = FileNotFoundError
            raise error_class("missing font metrics file: %s" % fontname)
        if c != 0 and tfm.checksum != 0 and c != tfm.checksum:
            raise ValueError('tfm checksum mismatch: %s'%n)

        vf = _vffile(fontname)

        self.fonts[k] = DviFont(scale=s, tfm=tfm, texname=n, vf=vf)

    def _post(self):
        if self.state != _dvistate.outer:
            raise ValueError("misplaced post in dvi file")
        self.state = _dvistate.post_post
        # TODO: actually read the postamble and finale?
        # currently post_post just triggers closing the file

    def _post_post(self):
        raise NotImplementedError

class DviFont(object):
    """
    Object that holds a font's texname and size, supports comparison,
    and knows the widths of glyphs in the same units as the AFM file.
    There are also internal attributes (for use by dviread.py) that
    are *not* used for comparison.

    The size is in Adobe points (converted from TeX points).

    .. attribute:: texname

       Name of the font as used internally by TeX and friends. This
       is usually very different from any external font names, and
       :class:`dviread.PsfontsMap` can be used to find the external
       name of the font.

    .. attribute:: size

       Size of the font in Adobe points, converted from the slightly
       smaller TeX points.

    .. attribute:: widths

       Widths of glyphs in glyph-space units, typically 1/1000ths of
       the point size.

    """
    __slots__ = ('texname', 'size', 'widths', '_scale', '_vf', '_tfm')

    def __init__(self, scale, tfm, texname, vf):
        if six.PY3 and isinstance(texname, bytes):
            texname = texname.decode('ascii')
        self._scale, self._tfm, self.texname, self._vf = \
            scale, tfm, texname, vf
        self.size = scale * (72.0 / (72.27 * 2**16))
        try:
            nchars = max(six.iterkeys(tfm.width)) + 1
        except ValueError:
            nchars = 0
        self.widths = [ (1000*tfm.width.get(char, 0)) >> 20
                        for char in xrange(nchars) ]

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.texname == other.texname and self.size == other.size

    def __ne__(self, other):
        return not self.__eq__(other)

    def _width_of(self, char):
        """
        Width of char in dvi units. For internal use by dviread.py.
        """

        width = self._tfm.width.get(char, None)
        if width is not None:
            return _mul2012(width, self._scale)

        matplotlib.verbose.report(
            'No width for char %d in font %s' % (char, self.texname),
            'debug')
        return 0

    def _height_depth_of(self, char):
        """
        Height and depth of char in dvi units. For internal use by dviread.py.
        """

        result = []
        for metric,name in ((self._tfm.height, "height"),
                            (self._tfm.depth, "depth")):
            value = metric.get(char, None)
            if value is None:
                matplotlib.verbose.report(
                    'No %s for char %d in font %s' % (name, char, self.texname),
                    'debug')
                result.append(0)
            else:
                result.append(_mul2012(value, self._scale))
        return result

class Vf(Dvi):
    """
    A virtual font (\*.vf file) containing subroutines for dvi files.

    Usage::

      vf = Vf(filename)
      glyph = vf[code]
      glyph.text, glyph.boxes, glyph.width
    """

    def __init__(self, filename):
        Dvi.__init__(self, filename, 0)
        try:
            self._first_font = None
            self._chars = {}
            self._packet_ends = None
            self._read()
        finally:
            self.close()

    def __getitem__(self, code):
        return self._chars[code]

    def _dispatch(self, byte):
        # If we are in a packet, execute the dvi instructions
        if self.state == _dvistate.inpage:
            byte_at = self.file.tell()-1
            if byte_at == self._packet_ends:
                self._finalize_packet()
                # fall through
            elif byte_at > self._packet_ends:
                raise ValueError("Packet length mismatch in vf file")
            else:
                if byte in (139, 140) or byte >= 243:
                    raise ValueError("Inappropriate opcode %d in vf file" % byte)
                Dvi._dispatch(self, byte)
                return

        # We are outside a packet
        if byte < 242:          # a short packet (length given by byte)
            cc, tfm = self._arg(1), self._arg(3)
            self._init_packet(byte, cc, tfm)
        elif byte == 242:       # a long packet
            pl, cc, tfm = [ self._arg(x) for x in (4, 4, 4) ]
            self._init_packet(pl, cc, tfm)
        elif 243 <= byte <= 246:
            Dvi._dispatch(self, byte)
        elif byte == 247:       # preamble
            i, k = self._arg(1), self._arg(1)
            x = self.file.read(k)
            cs, ds = self._arg(4), self._arg(4)
            self._pre(i, x, cs, ds)
        elif byte == 248:       # postamble (just some number of 248s)
            self.state = _dvistate.post_post
        else:
            raise ValueError("unknown vf opcode %d" % byte)

    def _init_packet(self, pl, cc, tfm):
        if self.state != _dvistate.outer:
            raise ValueError("Misplaced packet in vf file")
        self.state = _dvistate.inpage
        self._packet_ends = self.file.tell() + pl
        self._packet_char = cc
        self._packet_width = tfm
        self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
        self.stack, self.text, self.boxes = [], [], []
        self.f = self._first_font

    def _finalize_packet(self):
        self._chars[self._packet_char] = mpl_cbook.Bunch(
            text=self.text, boxes=self.boxes, width = self._packet_width)
        self.state = _dvistate.outer

    def _pre(self, i, x, cs, ds):
        if self.state != _dvistate.pre:
            raise ValueError("pre command in middle of vf file")
        if i != 202:
            raise ValueError("Unknown vf format %d" % i)
        if len(x):
            matplotlib.verbose.report('vf file comment: ' + x, 'debug')
        self.state = _dvistate.outer
        # cs = checksum, ds = design size

    def _fnt_def(self, k, *args):
        Dvi._fnt_def(self, k, *args)
        if self._first_font is None:
            self._first_font = k

def _fix2comp(num):
    """
    Convert from two's complement to negative.
    """
    assert 0 <= num < 2**32
    if num & 2**31:
        return num - 2**32
    else:
        return num

def _mul2012(num1, num2):
    """
    Multiply two numbers in 20.12 fixed point format.
    """
    # Separated into a function because >> has surprising precedence
    return (num1*num2) >> 20

class Tfm(object):
    """
    A TeX Font Metric file. This implementation covers only the bare
    minimum needed by the Dvi class.

    .. attribute:: checksum

       Used for verifying against the dvi file.

    .. attribute:: design_size

       Design size of the font (in what units?)

    .. attribute::  width

       Width of each character, needs to be scaled by the factor
       specified in the dvi file. This is a dict because indexing may
       not start from 0.

    .. attribute:: height

       Height of each character.

    .. attribute:: depth

       Depth of each character.
    """
    __slots__ = ('checksum', 'design_size', 'width', 'height', 'depth')

    def __init__(self, filename):
        matplotlib.verbose.report('opening tfm file ' + filename, 'debug')
        with open(filename, 'rb') as file:
            header1 = file.read(24)
            lh, bc, ec, nw, nh, nd = \
                struct.unpack(str('!6H'), header1[2:14])
            matplotlib.verbose.report(
                'lh=%d, bc=%d, ec=%d, nw=%d, nh=%d, nd=%d' % (
                    lh, bc, ec, nw, nh, nd), 'debug')
            header2 = file.read(4*lh)
            self.checksum, self.design_size = \
                struct.unpack(str('!2I'), header2[:8])
            # there is also encoding information etc.
            char_info = file.read(4*(ec-bc+1))
            widths = file.read(4*nw)
            heights = file.read(4*nh)
            depths = file.read(4*nd)

        self.width, self.height, self.depth = {}, {}, {}
        widths, heights, depths = \
            [ struct.unpack(str('!%dI') % (len(x)/4), x)
              for x in (widths, heights, depths) ]
        for idx, char in enumerate(xrange(bc, ec+1)):
            self.width[char] = _fix2comp(widths[ord(char_info[4*idx])])
            self.height[char] = _fix2comp(heights[ord(char_info[4*idx+1]) >> 4])
            self.depth[char] = _fix2comp(depths[ord(char_info[4*idx+1]) & 0xf])

class PsfontsMap(object):
    """
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.
    Usage::

     >>> map = PsfontsMap(find_tex_file('pdftex.map'))
     >>> entry = map['ptmbo8r']
     >>> entry.texname
     'ptmbo8r'
     >>> entry.psname
     'Times-Bold'
     >>> entry.encoding
     '/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
     >>> entry.effects
     {'slant': 0.16700000000000001}
     >>> entry.filename

    For historical reasons, TeX knows many Type-1 fonts by different
    names than the outside world. (For one thing, the names have to
    fit in eight characters.) Also, TeX's native fonts are not Type-1
    but Metafont, which is nontrivial to convert to PostScript except
    as a bitmap. While high-quality conversions to Type-1 format exist
    and are shipped with modern TeX distributions, we need to know
    which Type-1 fonts are the counterparts of which native fonts. For
    these reasons a mapping is needed from internal font names to font
    file names.

    A texmf tree typically includes mapping files called e.g.
    psfonts.map, pdftex.map, dvipdfm.map. psfonts.map is used by
    dvips, pdftex.map by pdfTeX, and dvipdfm.map by dvipdfm.
    psfonts.map might avoid embedding the 35 PostScript fonts (i.e.,
    have no filename for them, as in the Times-Bold example above),
    while the pdf-related files perhaps only avoid the "Base 14" pdf
    fonts. But the user may have configured these files differently.
    """
    __slots__ = ('_font',)

    def __init__(self, filename):
        self._font = {}
        with open(filename, 'rt') as file:
            self._parse(file)

    def __getitem__(self, texname):
        try:
            result = self._font[texname]
        except KeyError:
            result = self._font[texname.decode('ascii')]
        fn, enc = result.filename, result.encoding
        if fn is not None and not fn.startswith('/'):
            result.filename = find_tex_file(fn)
        if enc is not None and not enc.startswith('/'):
            result.encoding = find_tex_file(result.encoding)
        return result

    def _parse(self, file):
        """Parse each line into words."""
        for line in file:
            line = line.strip()
            if line == '' or line.startswith('%'):
                continue
            words, pos = [], 0
            while pos < len(line):
                if line[pos] == '"': # double quoted word
                    pos += 1
                    end = line.index('"', pos)
                    words.append(line[pos:end])
                    pos = end + 1
                else:                # ordinary word
                    end = line.find(' ', pos+1)
                    if end == -1: end = len(line)
                    words.append(line[pos:end])
                    pos = end
                while pos < len(line) and line[pos] == ' ':
                    pos += 1
            self._register(words)

    def _register(self, words):
        """Register a font described by "words".

        The format is, AFAIK: texname fontname [effects and filenames]
        Effects are PostScript snippets like ".177 SlantFont",
        filenames begin with one or two less-than signs. A filename
        ending in enc is an encoding file, other filenames are font
        files. This can be overridden with a left bracket: <[foobar
        indicates an encoding file named foobar.

        There is some difference between <foo.pfb and <<bar.pfb in
        subsetting, but I have no example of << in my TeX installation.
        """

        # If the map file specifies multiple encodings for a font, we
        # follow pdfTeX in choosing the last one specified. Such
        # entries are probably mistakes but they have occurred.
        # http://tex.stackexchange.com/questions/10826/
        # http://article.gmane.org/gmane.comp.tex.pdftex/4914

        texname, psname = words[:2]
        effects, encoding, filename = '', None, None
        for word in words[2:]:
            if not word.startswith('<'):
                effects = word
            else:
                word = word.lstrip('<')
                if word.startswith('[') or word.endswith('.enc'):
                    if encoding is not None:
                        matplotlib.verbose.report(
                            'Multiple encodings for %s = %s'
                            % (texname, psname), 'debug')
                    if word.startswith('['):
                        encoding = word[1:]
                    else:
                        encoding = word
                else:
                    assert filename is None
                    filename = word

        eff = effects.split()
        effects = {}
        try:
            effects['slant'] = float(eff[eff.index('SlantFont')-1])
        except ValueError:
            pass
        try:
            effects['extend'] = float(eff[eff.index('ExtendFont')-1])
        except ValueError:
            pass

        self._font[texname] = mpl_cbook.Bunch(
            texname=texname, psname=psname, effects=effects,
            encoding=encoding, filename=filename)

class Encoding(object):
    """
    Parses a \*.enc file referenced from a psfonts.map style file.
    The format this class understands is a very limited subset of
    PostScript.

    Usage (subject to change)::

      for name in Encoding(filename):
          whatever(name)
    """
    __slots__ = ('encoding',)

    def __init__(self, filename):
        with open(filename, 'rt') as file:
            matplotlib.verbose.report('Parsing TeX encoding ' + filename, 'debug-annoying')
            self.encoding = self._parse(file)
            matplotlib.verbose.report('Result: ' + repr(self.encoding), 'debug-annoying')

    def __iter__(self):
        for name in self.encoding:
            yield name

    def _parse(self, file):
        result = []

        state = 0
        for line in file:
            comment_start = line.find('%')
            if comment_start > -1:
                line = line[:comment_start]
            line = line.strip()

            if state == 0:
                # Expecting something like /FooEncoding [
                if '[' in line:
                    state = 1
                    line = line[line.index('[')+1:].strip()

            if state == 1:
                if ']' in line: # ] def
                    line = line[:line.index(']')]
                    state = 2
                words = line.split()
                for w in words:
                    if w.startswith('/'):
                        # Allow for /abc/def/ghi
                        subwords = w.split('/')
                        result.extend(subwords[1:])
                    else:
                        raise ValueError("Broken name in encoding file: " + w)

        return result

def find_tex_file(filename, format=None):
    """
    Call :program:`kpsewhich` to find a file in the texmf tree. If
    *format* is not None, it is used as the value for the
    `--format` option.

    Apparently most existing TeX distributions on Unix-like systems
    use kpathsea. I hear MikTeX (a popular distribution on Windows)
    doesn't use kpathsea, so what do we do? (TODO)

    .. seealso::

      `Kpathsea documentation <http://www.tug.org/kpathsea/>`_
        The library that :program:`kpsewhich` is part of.
    """

    cmd = ['kpsewhich']
    if format is not None:
        cmd += ['--format=' + format]
    cmd += [filename]

    matplotlib.verbose.report('find_tex_file(%s): %s' \
                                  % (filename,cmd), 'debug')
    # stderr is unused, but reading it avoids a subprocess optimization
    # that breaks EINTR handling in some Python versions:
    # http://bugs.python.org/issue12493
    # https://github.com/matplotlib/matplotlib/issues/633
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    result = pipe.communicate()[0].rstrip()
    matplotlib.verbose.report('find_tex_file result: %s' % result,
                              'debug')
    return result.decode('ascii')

# With multiple text objects per figure (e.g., tick labels) we may end
# up reading the same tfm and vf files many times, so we implement a
# simple cache. TODO: is this worth making persistent?

_tfmcache = {}
_vfcache = {}

def _fontfile(texname, class_, suffix, cache):
    try:
        return cache[texname]
    except KeyError:
        pass

    filename = find_tex_file(texname + suffix)
    if filename:
        result = class_(filename)
    else:
        result = None

    cache[texname] = result
    return result

def _tfmfile(texname):
    return _fontfile(texname, Tfm, '.tfm', _tfmcache)

def _vffile(texname):
    return _fontfile(texname, Vf, '.vf', _vfcache)



if __name__ == '__main__':
    import sys
    matplotlib.verbose.set_level('debug-annoying')
    fname = sys.argv[1]
    try: dpi = float(sys.argv[2])
    except IndexError: dpi = None
    dvi = Dvi(fname, dpi)
    fontmap = PsfontsMap(find_tex_file('pdftex.map'))
    for page in dvi:
        print('=== new page ===')
        fPrev = None
        for x,y,f,c,w in page.text:
            if f != fPrev:
                print('font', f.texname, 'scaled', f._scale/pow(2.0,20))
                fPrev = f
            print(x,y,c, 32 <= c < 128 and chr(c) or '.', w)
        for x,y,w,h in page.boxes:
            print(x,y,'BOX',w,h)
