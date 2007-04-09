"""
An experimental module for reading single-page dvi files output by
TeX. Several limitations make this not (currently) useful as a
general-purpose dvi preprocessor. The idea is that the file has a
single page with only a single formula or other piece of text.

Interface:

   dvi = Dvi(filename)
   dvi.read()
   text, boxes = dvi.output(72)
   for x,y,font,glyph in text:
       fontname, pointsize = dvi.fontinfo(font)
       ...
   for x,y,height,width in boxes:
       ...
"""

from matplotlib.cbook import Bunch
import os
import struct

dvistate = Bunch(pre=0, outer=1, inpage=2, post_post=3, finale=4)

class Dvi(object):

    def __init__(self, filename):
        self.filename = filename
        self.text = []          # list of (x,y,fontnum,glyphnum)
        self.boxes = []         # list of (x,y,width,height)
        self.fonts = {}

    def output(self, dpi):
        """Return lists of text and box objects transformed into a standard
        Cartesian coordinate system at the given dpi value. The coordinates
        are floating point numbers, but otherwise precision is not lost and
        coordinate values are not clipped to integers."""
        t0 = self.text[0]
        minx, miny, maxx, maxy = t0[0], t0[1], t0[0], t0[1]
        for x,y,_,_ in self.text + self.boxes:
            if x < minx: minx = x
            if y < miny: miny = y
            if x > maxx: maxx = x
            if y > maxy: maxy = y
        d = dpi / (72.27 * 2**16) # from TeX's "scaled points" to dpi units
        text =  [ ((x-minx)*d, (maxy-y)*d, f, g) for (x,y,f,g) in self.text ]
        boxes = [ ((x-minx)*d, (maxy-y)*d, h*d, w*d) for (x,y,h,w) in self.boxes ]
        return text, boxes

    def fontinfo(self, f):
        """Name and size in (Adobe) points."""
        return self.fonts[f].name, self.fonts[f].scale * (72.0 / (72.27 * 2**16))

    def read(self, debug=False):
        self.file = open(self.filename, 'rb')
        try:
            self.state = dvistate.pre
            while True:
                byte = ord(self.file.read(1))
                if byte == '':
                    break           # eof
                self.dispatch(byte)
                if debug and self.state == dvistate.inpage:
                    print self.h, self.v
                if byte == 140: break # end of page; we only read a single page for now
        finally:
            self.file.close()

    def arg(self, nbytes, signed=False):
        str = self.file.read(nbytes)
        value = ord(str[0])
        if signed and value >= 0x80:
            value = value - 0x100
        for i in range(1, nbytes):
            value = 0x100*value + ord(str[i])
        return value

    def dispatch(self, byte):
        if 0 <= byte <= 127: self.set_char(byte)
        elif byte == 128: self.set_char(self.arg(1))
        elif byte == 129: self.set_char(self.arg(2))
        elif byte == 130: self.set_char(self.arg(3))
        elif byte == 131: self.set_char(self.arg(4, True))
        elif byte == 132: self.set_rule(self.arg(4, True), self.arg(4, True))
        elif byte == 133: self.put_char(self.arg(1))
        elif byte == 134: self.put_char(self.arg(2))
        elif byte == 135: self.put_char(self.arg(3))
        elif byte == 136: self.put_char(self.arg(4, True))
        elif byte == 137: self.put_rule(self.arg(4, True), self.arg(4, True))
        elif byte == 138: self.nop()
        elif byte == 139: self.bop(*[self.arg(4, True) for i in range(11)])
        elif byte == 140: self.eop()
        elif byte == 141: self.push()
        elif byte == 142: self.pop()
        elif byte == 143: self.right(self.arg(1, True))
        elif byte == 144: self.right(self.arg(2, True))
        elif byte == 145: self.right(self.arg(3, True))
        elif byte == 146: self.right(self.arg(4, True))
        elif byte == 147: self.right_w(None)
        elif byte == 148: self.right_w(self.arg(1, True))
        elif byte == 149: self.right_w(self.arg(2, True))
        elif byte == 150: self.right_w(self.arg(3, True))
        elif byte == 151: self.right_w(self.arg(4, True))
        elif byte == 152: self.right_x(None)
        elif byte == 153: self.right_x(self.arg(1, True))
        elif byte == 154: self.right_x(self.arg(2, True))
        elif byte == 155: self.right_x(self.arg(3, True))
        elif byte == 156: self.right_x(self.arg(4, True))
        elif byte == 157: self.down(self.arg(1, True))
        elif byte == 158: self.down(self.arg(2, True))
        elif byte == 159: self.down(self.arg(3, True))
        elif byte == 160: self.down(self.arg(4, True))
        elif byte == 161: self.down_y(None)
        elif byte == 162: self.down_y(self.arg(1, True))
        elif byte == 163: self.down_y(self.arg(2, True))
        elif byte == 164: self.down_y(self.arg(3, True))
        elif byte == 165: self.down_y(self.arg(4, True))
        elif byte == 166: self.down_z(None)
        elif byte == 167: self.down_z(self.arg(1, True))
        elif byte == 168: self.down_z(self.arg(2, True))
        elif byte == 169: self.down_z(self.arg(3, True))
        elif byte == 170: self.down_z(self.arg(4, True))
        elif 171 <= byte <= 234: self.fnt_num(byte-171)
        elif byte == 235: self.fnt_num(self.arg(1))
        elif byte == 236: self.fnt_num(self.arg(2))
        elif byte == 237: self.fnt_num(self.arg(3))
        elif byte == 238: self.fnt_num(self.arg(4, True))
        elif 239 <= byte <= 242:
            len = self.arg(byte-238)
            special = self.file.read(len)
            self.xxx(special)
        elif 243 <= byte <= 246:
            k = self.arg(byte-242, byte==246)
            c, s, d, a, l = [ self.arg(x) for x in (4, 4, 4, 1, 1) ]
            n = self.file.read(a+l)
            self.fnt_def(k, c, s, d, a, l, n)
        elif byte == 247: 
            i, num, den, mag, k = [ self.arg(x) for x in (1, 4, 4, 4, 1) ]
            x = self.file.read(k)
            self.pre(i, num, den, mag, x)
        elif byte == 248: self.post()
        elif byte == 249: self.post_post()
        else:
            raise ValueError, "unknown command: byte %d"%byte

    def pre(self, i, num, den, mag, comment):
        if self.state != dvistate.pre: 
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
        self.state = dvistate.outer

    def set_char(self, char):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced set_char in dvi file"
        self.put_char(char)
        font = self.fonts[self.f]
        width = font.tfm.width[char]
        width = (width * font.scale) >> 20
        self.h += width

    def set_rule(self, a, b):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced set_rule in dvi file"
        self.put_rule(a, b)
        self.h += b

    def put_char(self, char):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced put_char in dvi file"
        self.text.append((self.h, self.v, self.f, char))

    def put_rule(self, a, b):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced put_rule in dvi file"
        if a > 0 and b > 0:
            self.boxes.append((self.h, self.v, a, b))

    def nop(self):
        pass

    def bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
        if self.state != dvistate.outer:
            raise ValueError, "misplaced bop in dvi file"
        self.state = dvistate.inpage
        self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
        self.stack = []

    def eop(self):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced eop in dvi file"
        self.state = dvistate.outer
        del self.h, self.v, self.w, self.x, self.y, self.z, self.stack

    def push(self):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced push in dvi file"
        self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))

    def pop(self):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced pop in dvi file"
        self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()

    def right(self, b):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced right in dvi file"
        self.h += b

    def right_w(self, new_w):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced w in dvi file"
        if new_w is not None:
            self.w = new_w
        self.h += self.w

    def right_x(self, new_x):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced x in dvi file"
        if new_x is not None:
            self.x = new_x
        self.h += self.x

    def down(self, a):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced down in dvi file"
        self.v += a

    def down_y(self, new_y):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced y in dvi file"
        if new_y is not None:
            self.y = new_y
        self.v += self.y

    def down_z(self, new_z):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced z in dvi file"
        if new_z is not None:
            self.z = new_z
        self.v += self.z

    def fnt_num(self, k):
        if self.state != dvistate.inpage:
            raise ValueError, "misplaced fnt_num in dvi file"
        self.f = k

    def xxx(self, special):
        pass

    def fnt_def(self, k, c, s, d, a, l, n):
        filename = n[-l:] + '.tfm'
        pipe = os.popen('kpsewhich ' + filename, 'r')
        filename = pipe.readline().rstrip()
        pipe.close()
        tfm = Tfm(filename)
        if c != 0 and tfm.checksum != 0 and c != tfm.checksum:
            raise ValueError, 'tfm checksum mismatch: %s'%n
        # It seems that the assumption behind the following check is incorrect:
        #if d != tfm.design_size:
        #    raise ValueError, 'tfm design size mismatch: %d in dvi, %d in %s'%\
        #        (d, tfm.design_size, n)
        self.fonts[k] = Bunch(scale=s, tfm=tfm, name=n)

    def post(self):
        raise NotImplementedError

    def post_post(self):
        raise NotImplementedError

class Tfm(object):

    def __init__(self, filename):
        file = open(filename, 'rb')

        header1 = file.read(24)
        lh, bc, ec, nw = \
            struct.unpack('!4H', header1[2:10])
        header2 = file.read(4*lh)
        self.checksum, self.design_size = \
            struct.unpack('!2I', header2[:8])
        # plus encoding information etc.

        char_info = file.read(4*(ec-bc+1))
        widths = file.read(4*nw)

        file.close()
        
        widths = struct.unpack('!%dI' % nw, widths)
        self.width = {}
        for i in range(ec-bc):
            self.width[bc+i] = widths[ord(char_info[4*i])]

if __name__ == '__main__':
    dvi = Dvi('foo.dvi')
    dvi.read(debug=True)
    for x,y,f,c in dvi.text:
        print x,y,c,chr(c),dvi.fonts[f].__dict__
    print dvi.output(72)

