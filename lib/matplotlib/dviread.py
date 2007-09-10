"""
An experimental module for reading dvi files output by TeX. Several
limitations make this not (currently) useful as a general-purpose dvi
preprocessor.

Interface:

    dvi = Dvi(filename, 72)
    for page in dvi:          # iterate over pages
        w, h, d = page.width, page.height, page.descent
        for x,y,font,glyph,width in page.text:
            fontname, pointsize = dvi.fontinfo(font)
            ...
        for x,y,height,width in page.boxes:
            ...
"""

# TODO: support TeX virtual fonts (*.vf) which are a sort of
#       subroutine collections for dvi files

import matplotlib
import matplotlib.cbook as mpl_cbook
import numpy as npy
import os
import struct

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

    def __iter__(self):
        """
        Iterate through the pages of the file.

        Returns (text, pages) pairs, where:
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
        minx, miny, maxx, maxy = npy.inf, npy.inf, -npy.inf, -npy.inf
        maxy_pure = -npy.inf
        for elt in self.text + self.boxes:
            if len(elt) == 4:   # box
                x,y,h,w = elt
                e = 0           # zero depth
            else:               # glyph
                x,y,f,g,w = elt
                font = self.fonts[f]
                h = (font.scale * font.tfm.height[g]) >> 20
                e = (font.scale * font.tfm.depth[g]) >> 20
            minx = min(minx, x)
            miny = min(miny, y - h)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + e)
            maxy_pure = max(maxy_pure, y)

        d = self.dpi / (72.27 * 2**16) # from TeX's "scaled points" to dpi units
        text =  [ ((x-minx)*d, (maxy-y)*d, f, g, w*d) for (x,y,f,g,w) in self.text ]
        boxes = [ ((x-minx)*d, (maxy-y)*d, h*d, w*d) for (x,y,h,w) in self.boxes ]

        return mpl_cbook.Bunch(text=text, boxes=boxes, 
                               width=(maxx-minx)*d, 
                               height=(maxy_pure-miny)*d, 
                               descent=(maxy-maxy_pure)*d)

    def fontinfo(self, f):
        """
        texname, pointsize = dvi.fontinfo(fontnum)

        Name and size in points (Adobe points, not TeX points).
        """
        return self.fonts[f].name, self.fonts[f].scale * (72.0 / (72.27 * 2**16))

    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        while True:
            byte = ord(self.file.read(1))
            self._dispatch(byte)
            if self.state == _dvistate.inpage:
                matplotlib.verbose.report(
                    'Dvi._read: after %d at %f,%f' % 
                    (byte, self.h, self.v), 
                    'debug-annoying')
            if byte == 140: # end of page
                return True
            if self.state == _dvistate.post_post: # end of file
                self.close()
                return False

    def _arg(self, nbytes, signed=False):
        """
        Read and return an integer argument "nbytes" long.
        Signedness is determined by the "signed" keyword.
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
        Based on the opcode "byte", read the correct kinds of
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
            raise ValueError, "unknown command: byte %d"%byte

    def _pre(self, i, num, den, mag, comment):
        if self.state != _dvistate.pre:
            raise ValueError, "pre command in middle of dvi file"
        if i != 2:
            raise ValueError, "Unknown dvi format %d"%i
        if num != 25400000 or den != 7227 * 2**16:
            raise ValueError, "nonstandard units in dvi file"
            # meaning: TeX always uses those exact values, so it
            # should be enough for us to support those
            # (There are 72.27 pt to an inch so 7227 pt =
            # 7227 * 2**16 sp to 100 in. The numerator is multiplied
            # by 10^5 to get units of 10**-7 meters.)
        if mag != 1000:
            raise ValueError, "nonstandard magnification in dvi file"
            # meaning: LaTeX seems to frown on setting \mag, so
            # I think we can assume this is constant
        self.state = _dvistate.outer

    def _width_of(self, char):
        font = self.fonts[self.f]
        width = font.tfm.width[char]
        width = (width * font.scale) >> 20
        return width

    def _set_char(self, char):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced set_char in dvi file"
        self._put_char(char)
        self.h += self._width_of(char)

    def _set_rule(self, a, b):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced set_rule in dvi file"
        self._put_rule(a, b)
        self.h += b

    def _put_char(self, char):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced put_char in dvi file"
        self.text.append((self.h, self.v, self.f, char, self._width_of(char)))

    def _put_rule(self, a, b):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced put_rule in dvi file"
        if a > 0 and b > 0:
            self.boxes.append((self.h, self.v, a, b))

    def _nop(self):
        pass

    def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
        if self.state != _dvistate.outer:
            print '+++', self.state
            raise ValueError, "misplaced bop in dvi file"
        self.state = _dvistate.inpage
        self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
        self.stack = []
        self.text = []          # list of (x,y,fontnum,glyphnum)
        self.boxes = []         # list of (x,y,width,height)

    def _eop(self):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced eop in dvi file"
        self.state = _dvistate.outer
        del self.h, self.v, self.w, self.x, self.y, self.z, self.stack

    def _push(self):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced push in dvi file"
        self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))

    def _pop(self):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced pop in dvi file"
        self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()

    def _right(self, b):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced right in dvi file"
        self.h += b

    def _right_w(self, new_w):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced w in dvi file"
        if new_w is not None:
            self.w = new_w
        self.h += self.w

    def _right_x(self, new_x):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced x in dvi file"
        if new_x is not None:
            self.x = new_x
        self.h += self.x

    def _down(self, a):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced down in dvi file"
        self.v += a

    def _down_y(self, new_y):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced y in dvi file"
        if new_y is not None:
            self.y = new_y
        self.v += self.y

    def _down_z(self, new_z):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced z in dvi file"
        if new_z is not None:
            self.z = new_z
        self.v += self.z

    def _fnt_num(self, k):
        if self.state != _dvistate.inpage:
            raise ValueError, "misplaced fnt_num in dvi file"
        self.f = k

    def _xxx(self, special):
        matplotlib.verbose.report(
            'Dvi._xxx: encountered special: %s'
            % ''.join((32 <= ord(ch) < 127) and ch 
                      or '<%02x>' % ord(ch)
                      for ch in special),
            'debug')

    def _fnt_def(self, k, c, s, d, a, l, n):
        filename = find_tex_file(n[-l:] + '.tfm')
        tfm = Tfm(filename)
        if c != 0 and tfm.checksum != 0 and c != tfm.checksum:
            raise ValueError, 'tfm checksum mismatch: %s'%n
        # It seems that the assumption behind the following check is incorrect:
        #if d != tfm.design_size:
        #    raise ValueError, 'tfm design size mismatch: %d in dvi, %d in %s'%\
        #        (d, tfm.design_size, n)
        self.fonts[k] = mpl_cbook.Bunch(scale=s, tfm=tfm, name=n)

    def _post(self):
        if self.state != _dvistate.outer:
            raise ValueError, "misplaced post in dvi file"
        self.state = _dvistate.post_post
        # TODO: actually read the postamble and finale?
        # currently post_post just triggers closing the file

    def _post_post(self):
        raise NotImplementedError

class Tfm(object):
    """
    A TeX Font Metric file. This implementation covers only the bare
    minimum needed by the Dvi class.

    Attributes:
      checksum: for verifying against dvi file
      design_size: design size of the font (in what units?)
      width[i]: width of character #i, needs to be scaled 
        by the factor specified in the dvi file
        (this is a dict because indexing may not start from 0)
      height[i], depth[i]: height and depth of character #i
    """

    def __init__(self, filename):
        file = open(filename, 'rb')

        try:
            header1 = file.read(24)
            lh, bc, ec, nw, nh, nd = \
                struct.unpack('!6H', header1[2:14])
            header2 = file.read(4*lh)
            self.checksum, self.design_size = \
                struct.unpack('!2I', header2[:8])
            # there is also encoding information etc.
            char_info = file.read(4*(ec-bc+1))
            widths = file.read(4*nw)
            heights = file.read(4*nh)
            depths = file.read(4*nd)
        finally:
            file.close()

        self.width, self.height, self.depth = {}, {}, {}
        widths, heights, depths = \
            [ struct.unpack('!%dI' % n, x) 
              for n,x in [(nw, widths), (nh, heights), (nd, depths)] ]
        for i in range(ec-bc):
            self.width[bc+i] = widths[ord(char_info[4*i])]
            self.height[bc+i] = heights[ord(char_info[4*i+1]) >> 4]
            self.depth[bc+i] = depths[ord(char_info[4*i+1]) & 0xf]


class PsfontsMap(object):
    """
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.
    Usage: map = PsfontsMap('.../psfonts.map'); map['cmr10']

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
    psfonts.map, pdftex.map,  dvipdfm.map. psfonts.map is used by
    dvips, pdftex.map by pdfTeX, and dvipdfm.map by dvipdfm.
    psfonts.map might avoid embedding the 35 PostScript fonts, while
    the pdf-related files perhaps only avoid the "Base 14" pdf fonts.
    But the user may have configured these files differently.
    """
    
    def __init__(self, filename):
        self._font = {}
        file = open(filename, 'rt')
        try:
            self._parse(file)
        finally:
            file.close()

    def __getitem__(self, texname):
        result = self._font[texname]
        fn, enc = result.filename, result.encoding
        if fn is not None and not fn.startswith('/'):
            result.filename = find_tex_file(fn)
            result.afm = find_tex_file(fn[:-4] + '.afm')
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
        texname, psname = words[:2]
        effects, encoding, filename = [], None, None
        for word in words[2:]:
            if not word.startswith('<'):
                effects.append(word)
            else:
                word = word.lstrip('<')
                if word.startswith('['):
                    assert encoding is None
                    encoding = word[1:]
                elif word.endswith('.enc'):
                    assert encoding is None
                    encoding = word
                else:
                    assert filename is None
                    filename = word
        self._font[texname] = mpl_cbook.Bunch(
            texname=texname, psname=psname, effects=effects, 
            encoding=encoding, filename=filename)

class Encoding(object):

    def __init__(self, filename):
        file = open(filename, 'rt')
        try:
            self.encoding = self._parse(file)
        finally:
            file.close()

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
                    line = line[line.index('[')+1].strip()

            if state == 1:
                words = line.split()
                for w in words:
                    if w.startswith('/'):
                        # Allow for /abc/def/ghi
                        subwords = w.split('/')
                        result.extend(subwords[1:])
                    else:
                        raise ValueError, "Broken name in encoding file: " + w
                        
                # Expecting ] def
                if ']' in line:
                    break

        return result

def find_tex_file(filename, format=None):
    """
    Call kpsewhich to find a file in the texmf tree.
    If format is not None, it is used as the value for the --format option.
    See the kpathsea documentation for more information.

    Apparently most existing TeX distributions on Unix-like systems
    use kpathsea. I hear MikTeX (a popular distribution on Windows)
    doesn't use kpathsea, so what do we do? (TODO)
    """

    cmd = 'kpsewhich '
    if format is not None:
        assert "'" not in format
        cmd += "--format='" + format + "' "
    assert "'" not in filename
    cmd += "'" + filename + "'"

    pipe = os.popen(cmd, 'r')
    result = pipe.readline().rstrip()
    pipe.close()

    return result

if __name__ == '__main__':
    matplotlib.verbose.set_level('debug')
    dvi = Dvi('foo.dvi', 72)
    fontmap = PsfontsMap(find_tex_file('pdftex.map'))
    for text,boxes in dvi:
        print '=== new page ==='
        fPrev = None
        for x,y,f,c in text:
            texname = dvi.fonts[f].name
            print x,y,c,chr(c),texname
            if f != fPrev:
                print 'font', texname, '=', fontmap[texname].__dict__
                fPrev = f
        for x,y,w,h in boxes:
            print x,y,'BOX',w,h


