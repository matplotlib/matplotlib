"""
A class representing a Type 1 font.

This version merely reads pfa and pfb files and splits them for
embedding in pdf files. There is no support yet for subsetting or
anything like that.

Usage (subject to change):

   font = Type1Font(filename)
   clear_part, encrypted_part, finale = font.parts

Source: Adobe Technical Note #5040, Supporting Downloadable PostScript
Language Fonts.

If extending this class, see also: Adobe Type 1 Font Format, Adobe
Systems Incorporated, third printing, v1.1, 1993. ISBN 0-201-57044-0.
"""

import re
import struct

class Type1Font(object):

    def __init__(self, filename):
        file = open(filename, 'rb')
        try:
            data = self._read(file)
        finally:
            file.close()
        self.parts = self._split(data)
        #self._parse()

    def _read(self, file):
        rawdata = file.read()
        if not rawdata.startswith(chr(128)):
            return rawdata

        data = ''
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
                data += segment
            elif type == 2:     # binary data: encode in hexadecimal
                data += ''.join(['%02x' % ord(char)
                                      for char in segment])
            elif type == 3:     # end of file
                break
            else:
                raise RuntimeError, \
                    'Unknown segment type %d in pfb file' % type

        return data

    def _split(self, data):
        """
        Split the Type 1 font into its three main parts.

        The three parts are: (1) the cleartext part, which ends in a
        eexec operator; (2) the encrypted part; (3) the fixed part,
        which contains 512 ASCII zeros possibly divided on various
        lines, a cleartomark operator, and possibly something else.
        """

        # Cleartext part: just find the eexec and skip whitespace
        idx = data.index('eexec')
        idx += len('eexec')
        while data[idx] in ' \t\r\n':
            idx += 1
        len1 = idx

        # Encrypted part: find the cleartomark operator and count
        # zeros backward
        idx = data.rindex('cleartomark') - 1
        zeros = 512
        while zeros and data[idx] in ('0', '\n', '\r'):
            if data[idx] == '0':
                zeros -= 1
            idx -= 1
        if zeros:
            raise RuntimeError, 'Insufficiently many zeros in Type 1 font'

        # Convert encrypted part to binary (if we read a pfb file, we
        # may end up converting binary to hexadecimal to binary again;
        # but if we read a pfa file, this part is already in hex, and
        # I am not quite sure if even the pfb format guarantees that
        # it will be in binary).
        binary = ''.join([chr(int(data[i:i+2], 16))
                          for i in range(len1, idx, 2)])

        return data[:len1], binary, data[idx:]

    _whitespace = re.compile(r'[\0\t\r\014\n ]+')
    _delim = re.compile(r'[()<>[]{}/%]')
    _token = re.compile(r'/{0,2}[^]\0\t\r\v\n ()<>{}/%[]+')
    _comment = re.compile(r'%[^\r\n\v]*')
    _instring = re.compile(r'[()\\]')
    def _parse(self):
        """
        A very limited kind of parsing to find the Encoding of the
        font.
        """
        def tokens(text):
            """
            Yield pairs (position, token), ignoring comments and
            whitespace. Numbers count as tokens.
            """
            pos = 0
            while pos < len(text):
                match = self._comment.match(text[pos:]) or self._whitespace.match(text[pos:])
                if match:
                    pos += match.end()
                elif text[pos] == '(':
                    start = pos
                    pos += 1
                    depth = 1
                    while depth:
                        match = self._instring.search(text[pos:])
                        if match is None: return
                        if match.group() == '(':
                            depth += 1
                            pos += 1
                        elif match.group() == ')':
                            depth -= 1
                            pos += 1
                        else:
                            pos += 2
                    yield (start, text[start:pos])
                elif text[pos:pos+2] in ('<<', '>>'):
                    yield (pos, text[pos:pos+2])
                    pos += 2
                elif text[pos] == '<':
                    start = pos
                    pos += text[pos:].index('>')
                    yield (start, text[start:pos])
                else:
                    match = self._token.match(text[pos:])
                    if match:
                        yield (pos, match.group())
                        pos += match.end()
                    else:
                        yield (pos, text[pos])
                        pos += 1

        enc_starts, enc_ends = None, None
        state = 0
        # State transitions:
        # 0 -> /Encoding -> 1
        # 1 -> StandardEncoding -> 2 -> def -> (ends)
        # 1 -> dup -> 4 -> put -> 5
        # 5 -> dup -> 4 -> put -> 5
        # 5 -> def -> (ends)
        for pos,token in tokens(self.parts[0]):
            if state == 0 and token == '/Encoding':
                enc_starts = pos
                state = 1
            elif state == 1 and token == 'StandardEncoding':
                state = 2
            elif state in (2,5) and token == 'def':
                enc_ends = pos+3
                break
            elif state in (1,5) and token == 'dup':
                state = 4
            elif state == 4 and token == 'put':
                state = 5
        self.enc_starts, self.enc_ends = enc_starts, enc_ends
                
    
if __name__ == '__main__':
    import sys
    font = Type1Font(sys.argv[1])
    parts = font.parts
    print len(parts[0]), len(parts[1]), len(parts[2])
    #print parts[0][font.enc_starts:font.enc_ends]

