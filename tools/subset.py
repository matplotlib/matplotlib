#!/usr/bin/env python
#
# Copyright 2010-2012, Google Inc.
# Author: Mikhail Kashkin (mkashkin@gmail.com)
# Author: Raph Levien (<firstname.lastname>@gmail.com)
# Author: Dave Crossland (dave@understandinglimited.com)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Version 1.01 Released 2012-03-27
#
# A script for subsetting a font, using FontForge. See README for details.

# TODO 2013-04-08 ensure the menu files are as compact as possible by default,
# similar to subset.pl
# TODO 2013-05-22 in Arimo, the latin subset doesn't include ; but the greek
# does. why on earth is this happening?

import getopt
import os
import struct
import subprocess
import sys

import fontforge


def log_namelist(nam, unicode):
    if nam and isinstance(unicode, int):
        print(f"0x{unicode:04X}", fontforge.nameFromUnicode(unicode), file=nam)


def select_with_refs(font, unicode, newfont, pe=None, nam=None):
    newfont.selection.select(('more', 'unicode'), unicode)
    log_namelist(nam, unicode)
    if pe:
        print(f"SelectMore({unicode})", file=pe)
    try:
        for ref in font[unicode].references:
            newfont.selection.select(('more',), ref[0])
            log_namelist(nam, ref[0])
            if pe:
                print(f'SelectMore("{ref[0]}")', file=pe)
    except Exception:
        print(f'Resolving references on u+{unicode:04x} failed')


def subset_font_raw(font_in, font_out, unicodes, opts):
    if '--namelist' in opts:
        # 2010-12-06 DC To allow setting namelist filenames,
        # change getopt.gnu_getopt from namelist to namelist=
        # and invert comments on following 2 lines
        # nam_fn = opts['--namelist']
        nam_fn = f'{font_out}.nam'
        nam = open(nam_fn, 'w')
    else:
        nam = None
    if '--script' in opts:
        pe_fn = "/tmp/script.pe"
        pe = open(pe_fn, 'w')
    else:
        pe = None
    font = fontforge.open(font_in)
    if pe:
        print(f'Open("{font_in}")', file=pe)
        extract_vert_to_script(font_in, pe)
    for i in unicodes:
        select_with_refs(font, i, font, pe, nam)

    addl_glyphs = []
    if '--nmr' in opts:
        addl_glyphs.append('nonmarkingreturn')
    if '--null' in opts:
        addl_glyphs.append('.null')
    if '--nd' in opts:
        addl_glyphs.append('.notdef')
    for glyph in addl_glyphs:
        font.selection.select(('more',), glyph)
        if nam:
            print(f"0x{fontforge.unicodeFromName(glyph):0.4X}", glyph,
                  file=nam)
        if pe:
            print(f'SelectMore("{glyph}")', file=pe)

    flags = ()

    if '--opentype-features' in opts:
        flags += ('opentype',)

    if '--simplify' in opts:
        font.simplify()
        font.round()
        flags += ('omit-instructions',)

    if '--strip_names' in opts:
        font.sfnt_names = ()

    if '--new' in opts:
        font.copy()
        new = fontforge.font()
        new.encoding = font.encoding
        new.em = font.em
        new.layers['Fore'].is_quadratic = font.layers['Fore'].is_quadratic
        for i in unicodes:
            select_with_refs(font, i, new, pe, nam)
        new.paste()
        # This is a hack - it should have been taken care of above.
        font.selection.select('space')
        font.copy()
        new.selection.select('space')
        new.paste()
        new.sfnt_names = font.sfnt_names
        font = new
    else:
        font.selection.invert()
        print("SelectInvert()", file=pe)
        font.cut()
        print("Clear()", file=pe)

    if '--move-display' in opts:
        print("Moving display glyphs into unicode ranges...")
        font.familyname += " Display"
        font.fullname += " Display"
        font.fontname += "Display"
        font.appendSFNTName('English (US)', 'Family', font.familyname)
        font.appendSFNTName('English (US)', 16, font.familyname)
        font.appendSFNTName('English (US)', 17, 'Display')
        font.appendSFNTName('English (US)', 'Fullname', font.fullname)
        for glname in unicodes:
            font.selection.none()
            if isinstance(glname, str):
                if glname.endswith('.display'):
                    font.selection.select(glname)
                    font.copy()
                    font.selection.none()
                    newgl = glname.replace('.display', '')
                    font.selection.select(newgl)
                    font.paste()
                font.selection.select(glname)
                font.cut()

    if nam:
        print("Writing NameList", end="")
        nam.close()

    if pe:
        print(f'Generate("{font_out}")', file=pe)
        pe.close()
        subprocess.call(["fontforge", "-script", pe_fn])
    else:
        font.generate(font_out, flags=flags)
    font.close()

    if '--roundtrip' in opts:
        # FontForge apparently contains a bug where it incorrectly calculates
        # the advanceWidthMax in the hhea table, and a workaround is to open
        # and re-generate
        font2 = fontforge.open(font_out)
        font2.generate(font_out, flags=flags)


def subset_font(font_in, font_out, unicodes, opts):
    font_out_raw = font_out
    if not font_out_raw.endswith('.ttf'):
        font_out_raw += '.ttf'
    subset_font_raw(font_in, font_out_raw, unicodes, opts)
    if font_out != font_out_raw:
        os.rename(font_out_raw, font_out)
# 2011-02-14 DC this needs to only happen with --namelist is used
#        os.rename(font_out_raw + '.nam', font_out + '.nam')


def getsubset(subset, font_in):
    subsets = subset.split('+')

    quotes = [
        0x2013,  # endash
        0x2014,  # emdash
        0x2018,  # quoteleft
        0x2019,  # quoteright
        0x201A,  # quotesinglbase
        0x201C,  # quotedblleft
        0x201D,  # quotedblright
        0x201E,  # quotedblbase
        0x2022,  # bullet
        0x2039,  # guilsinglleft
        0x203A,  # guilsinglright
    ]

    latin = [
        *range(0x20, 0x7f),  # Basic Latin (A-Z, a-z, numbers)
        *range(0xa0, 0x100),  # Western European symbols and diacritics
        0x20ac,  # Euro
        0x0152,  # OE
        0x0153,  # oe
        0x003b,  # semicolon
        0x00b7,  # periodcentered
        0x0131,  # dotlessi
        0x02c6,  # circumflex
        0x02da,  # ring
        0x02dc,  # tilde
        0x2074,  # foursuperior
        0x2215,  # division slash
        0x2044,  # fraction slash
        0xe0ff,  # PUA: Font logo
        0xeffd,  # PUA: Font version number
        0xf000,  # PUA: font ppem size indicator: run
                 # `ftview -f 1255 10 Ubuntu-Regular.ttf` to see it in action!
    ]

    result = quotes

    if 'menu' in subset:
        font = fontforge.open(font_in)
        result = [
            *map(ord, font.familyname),
            0x0020,
        ]

    if 'latin' in subset:
        result += latin
    if 'latin-ext' in subset:
        # These ranges include Extended A, B, C, D, and Additional with the
        # exception of Vietnamese, which is a separate range
        result += [
            *range(0x100, 0x370),
            *range(0x1d00, 0x1ea0),
            *range(0x1ef2, 0x1f00),
            *range(0x2070, 0x20d0),
            *range(0x2c60, 0x2c80),
            *range(0xa700, 0xa800),
        ]
    if 'vietnamese' in subset:
        # 2011-07-16 DC: Charset from
        # http://vietunicode.sourceforge.net/charset/ + U+1ef9 from Fontaine
        result += [0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00C8, 0x00C9,
                   0x00CA, 0x00CC, 0x00CD, 0x00D2, 0x00D3, 0x00D4,
                   0x00D5, 0x00D9, 0x00DA, 0x00DD, 0x00E0, 0x00E1,
                   0x00E2, 0x00E3, 0x00E8, 0x00E9, 0x00EA, 0x00EC,
                   0x00ED, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F9,
                   0x00FA, 0x00FD, 0x0102, 0x0103, 0x0110, 0x0111,
                   0x0128, 0x0129, 0x0168, 0x0169, 0x01A0, 0x01A1,
                   0x01AF, 0x01B0, 0x20AB, *range(0x1EA0, 0x1EFA)]
    if 'greek' in subset:
        # Could probably be more aggressive here and exclude archaic
        # characters, but lack data
        result += [*range(0x370, 0x400)]
    if 'greek-ext' in subset:
        result += [*range(0x370, 0x400), *range(0x1f00, 0x2000)]
    if 'cyrillic' in subset:
        # Based on character frequency analysis
        result += [*range(0x400, 0x460), 0x490, 0x491, 0x4b0, 0x4b1, 0x2116]
    if 'cyrillic-ext' in subset:
        result += [
            *range(0x400, 0x530),
            0x20b4,
            # 0x2116 is the russian No, a number abbreviation similar to the
            # latin #, suggested by Alexei Vanyashin
            0x2116,
            *range(0x2de0, 0x2e00),
            *range(0xa640, 0xa6a0),
        ]
    if 'arabic' in subset:
        # Based on Droid Arabic Kufi 1.0
        result += [0x000D, 0x0020, 0x0621, 0x0627, 0x062D,
                   0x062F, 0x0631, 0x0633, 0x0635, 0x0637, 0x0639,
                   0x0643, 0x0644, 0x0645, 0x0647, 0x0648, 0x0649,
                   0x0640, 0x066E, 0x066F, 0x0660, 0x0661, 0x0662,
                   0x0663, 0x0664, 0x0665, 0x0666, 0x0667, 0x0668,
                   0x0669, 0x06F4, 0x06F5, 0x06F6, 0x06BE, 0x06D2,
                   0x06A9, 0x06AF, 0x06BA, 0x066A, 0x061F, 0x060C,
                   0x061B, 0x066B, 0x066C, 0x066D, 0x064B, 0x064D,
                   0x064E, 0x064F, 0x064C, 0x0650, 0x0651, 0x0652,
                   0x0653, 0x0654, 0x0655, 0x0670, 0x0656, 0x0615,
                   0x0686, 0x0623, 0x0625, 0x0622, 0x0671, 0x0628,
                   0x067E, 0x062A, 0x062B, 0x0679, 0x0629, 0x062C,
                   0x062E, 0x0630, 0x0688, 0x0632, 0x0691, 0x0698,
                   0x0634, 0x0636, 0x0638, 0x063A, 0x0641, 0x0642,
                   0x0646, 0x06D5, 0x06C0, 0x0624, 0x064A, 0x06CC,
                   0x06D3, 0x0626, 0x06C2, 0x06C1, 0x06C3, 0x06F0,
                   0x06F1, 0x06F2, 0x06F3, 0x06F9, 0x06F7, 0x06F8,
                   0xFC63, 0x0672, 0x0673, 0x0675, 0x0676, 0x0677,
                   0x0678, 0x067A, 0x067B, 0x067C, 0x067D, 0x067F,
                   0x0680, 0x0681, 0x0682, 0x0683, 0x0684, 0x0685,
                   0x0687, 0x0689, 0x068A, 0x068B, 0x068C, 0x068D,
                   0x068E, 0x068F, 0x0690, 0x0692, 0x0693, 0x0694,
                   0x0695, 0x0696, 0x0697, 0x0699, 0x069A, 0x069B,
                   0x069C, 0x069D, 0x069E, 0x069F, 0x06A0, 0x06A1,
                   0x06A2, 0x06A3, 0x06A5, 0x06A6, 0x06A7, 0x06A8,
                   0x06AA, 0x06AB, 0x06AC, 0x06AD, 0x06AE, 0x06B0,
                   0x06B1, 0x06B2, 0x06B3, 0x06B4, 0x06B5, 0x06B6,
                   0x06B7, 0x06B8, 0x06B9, 0x06BB, 0x06BC, 0x06BD,
                   0x06BF, 0x06C4, 0x06C5, 0x06CD, 0x06D6, 0x06D7,
                   0x06D8, 0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF,
                   0x06E1, 0x06E2, 0x06E3, 0x06E4, 0x06E5, 0x06E6,
                   0x06E7, 0x06E8, 0x06EA, 0x06EB, 0x06ED, 0x06FB,
                   0x06FC, 0x06FD, 0x06FE, 0x0600, 0x0601, 0x0602,
                   0x0603, 0x060E, 0x060F, 0x0610, 0x0611, 0x0612,
                   0x0613, 0x0614, 0x0657, 0x0658, 0x06EE, 0x06EF,
                   0x06FF, 0x060B, 0x061E, 0x0659, 0x065A, 0x065B,
                   0x065C, 0x065D, 0x065E, 0x0750, 0x0751, 0x0752,
                   0x0753, 0x0754, 0x0755, 0x0756, 0x0757, 0x0758,
                   0x0759, 0x075A, 0x075B, 0x075C, 0x075D, 0x075E,
                   0x075F, 0x0760, 0x0761, 0x0762, 0x0763, 0x0764,
                   0x0765, 0x0766, 0x0767, 0x0768, 0x0769, 0x076A,
                   0x076B, 0x076C, 0x076D, 0x06A4, 0x06C6, 0x06C7,
                   0x06C8, 0x06C9, 0x06CA, 0x06CB, 0x06CF, 0x06CE,
                   0x06D0, 0x06D1, 0x06D4, 0x06FA, 0x06DD, 0x06DE,
                   0x06E0, 0x06E9, 0x060D, 0xFD3E, 0xFD3F, 0x25CC,
                   # Added from
                   # https://groups.google.com/d/topic/googlefontdirectory-discuss/MwlMWMPNCXs/discussion
                   0x063b, 0x063c, 0x063d, 0x063e, 0x063f, 0x0620,
                   0x0674, 0x0674, 0x06EC]

    if 'dejavu-ext' in subset:
        # add all glyphnames ending in .display
        font = fontforge.open(font_in)
        for glyph in font.glyphs():
            if glyph.glyphname.endswith('.display'):
                result.append(glyph.glyphname)

    return result


# code for extracting vertical metrics from a TrueType font
class Sfnt:
    def __init__(self, data):
        version, numTables, _, _, _ = struct.unpack('>IHHHH', data[:12])
        self.tables = {}
        for i in range(numTables):
            tag, checkSum, offset, length = struct.unpack(
                '>4sIII', data[12 + 16 * i: 28 + 16 * i])
            self.tables[tag] = data[offset: offset + length]

    def hhea(self):
        r = {}
        d = self.tables['hhea']
        r['Ascender'], r['Descender'], r['LineGap'] = struct.unpack(
            '>hhh', d[4:10])
        return r

    def os2(self):
        r = {}
        d = self.tables['OS/2']
        r['fsSelection'], = struct.unpack('>H', d[62:64])
        r['sTypoAscender'], r['sTypoDescender'], r['sTypoLineGap'] = \
            struct.unpack('>hhh', d[68:74])
        r['usWinAscender'], r['usWinDescender'] = struct.unpack(
            '>HH', d[74:78])
        return r


def set_os2(pe, name, val):
    print(f'SetOS2Value("{name}", {val:d})', file=pe)


def set_os2_vert(pe, name, val):
    set_os2(pe, name + 'IsOffset', 0)
    set_os2(pe, name, val)


# Extract vertical metrics data directly out of font file, and emit
# script code to set the values in the generated font. This is a (rather
# ugly) workaround for the issue described in:
# https://sourceforge.net/p/fontforge/mailman/fontforge-users/thread/20100906085718.GB1907@khaled-laptop/
def extract_vert_to_script(font_in, pe):
    with open(font_in, 'rb') as in_file:
        data = in_file.read()
    sfnt = Sfnt(data)
    hhea = sfnt.hhea()
    os2 = sfnt.os2()
    set_os2_vert(pe, "WinAscent", os2['usWinAscender'])
    set_os2_vert(pe, "WinDescent", os2['usWinDescender'])
    set_os2_vert(pe, "TypoAscent", os2['sTypoAscender'])
    set_os2_vert(pe, "TypoDescent", os2['sTypoDescender'])
    set_os2_vert(pe, "HHeadAscent", hhea['Ascender'])
    set_os2_vert(pe, "HHeadDescent", hhea['Descender'])


def main(argv):
    optlist, args = getopt.gnu_getopt(argv, '', [
        'string=', 'strip_names', 'opentype-features', 'simplify', 'new',
        'script', 'nmr', 'roundtrip', 'subset=', 'namelist', 'null', 'nd',
        'move-display'])

    font_in, font_out = args
    opts = dict(optlist)
    if '--string' in opts:
        subset = map(ord, opts['--string'])
    else:
        subset = getsubset(opts.get('--subset', 'latin'), font_in)
    subset_font(font_in, font_out, subset, opts)


if __name__ == '__main__':
    main(sys.argv[1:])
