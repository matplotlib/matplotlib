"""
A class representing a Type 1 font.

This version merely allows reading in pfa and pfb files, and stores
the data in pfa format (which can be embedded in PostScript or PDF
files). A more complete class might support subsetting.

Usage:  font = Type1Font(filename)
        somefile.write(font.data) # writes out font in pfa format
        len1, len2, len3 = font.lengths() # needed for pdf embedding

Source: Adobe Technical Note #5040, Supporting Downloadable PostScript
Language Fonts.

If extending this class, see also: Adobe Type 1 Font Format, Adobe
Systems Incorporated, third printing, v1.1, 1993. ISBN 0-201-57044-0.
"""

import struct

class Type1Font(object):

    def __init__(self, filename):
        file = open(filename, 'rb')
        try:
            self._read(file)
        finally:
            file.close()

    def _read(self, file):
        rawdata = file.read()
        if not rawdata.startswith(chr(128)):
            self.data = rawdata
            return
        
        self.data = ''
        while len(rawdata) > 0:
            if not rawdata.startswith(chr(128)):
                raise RuntimeError, \
                    'Broken pfb file (expected byte 128, got %d)' % \
                    ord(rawdata[0])
            type = ord(rawdata[1])
            if type in (1,2):
                length, = struct.unpack('<i', rawdata[2:6])
                segment = rawdata[6:6+length]
                rawdata = rawdata[6+length:]

            if type == 1:       # ASCII text: include verbatim
                self.data += segment
            elif type == 2:     # binary data: encode in hexadecimal
                self.data += ''.join(['%02x' % ord(char)
                                      for char in segment])
            elif type == 3:     # end of file
                break
            else:
                raise RuntimeError, \
                    'Unknown segment type %d in pfb file' % type

    def lengths(self):
        """
        Compute the lengths of the three parts of a Type 1 font.

        The three parts are: (1) the cleartext part, which ends in a
        eexec operator; (2) the encrypted part; (3) the fixed part,
        which contains 512 ASCII zeros possibly divided on various
        lines, a cleartomark operator, and possibly something else.
        """

        # Cleartext part: just find the eexec and skip the eol char(s)
        idx = self.data.index('eexec')
        idx += len('eexec')
        while self.data[idx] in ('\n', '\r'):
            idx += 1
        len1 = idx

        # Encrypted part: find the cleartomark operator and count
        # zeros backward
        idx = self.data.rindex('cleartomark') - 1
        zeros = 512
        while zeros and self.data[idx] in ('0', '\n', '\r'):
            if self.data[idx] == '0':
                zeros -= 1
            idx -= 1
        if zeros:
            raise RuntimeError, 'Insufficiently many zeros in Type 1 font'

        len2 = idx - len1
        len3 = len(self.data) - idx

        return len1, len2, len3
            
if __name__ == '__main__':
    import sys
    font = Type1Font(sys.argv[1])
    sys.stdout.write(font.data)
