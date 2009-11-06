"""
This is a python interface to Adobe Font Metrics Files.  Although a
number of other python implementations exist (and may be more complete
than mine) I decided not to go with them because either they were
either

  1) copyrighted or used a non-BSD compatible license

  2) had too many dependencies and I wanted a free standing lib

  3) Did more than I needed and it was easier to write my own than
     figure out how to just get what I needed from theirs

It is pretty easy to use, and requires only built-in python libs::

    >>> from afm import AFM
    >>> fh = file('ptmr8a.afm')
    >>> afm = AFM(fh)
    >>> afm.string_width_height('What the heck?')
    (6220.0, 683)
    >>> afm.get_fontname()
    'Times-Roman'
    >>> afm.get_kern_dist('A', 'f')
    0
    >>> afm.get_kern_dist('A', 'y')
    -92.0
    >>> afm.get_bbox_char('!')
    [130, -9, 238, 676]
    >>> afm.get_bbox_font()
    [-168, -218, 1000, 898]


AUTHOR:
  John D. Hunter <jdh2358@gmail.com>
"""

import sys, os, re
from _mathtext_data import uni2type1

#Convert string the a python type

# some afm files have floats where we are expecting ints -- there is
# probably a better way to handle this (support floats, round rather
# than truncate).  But I don't know what the best approach is now and
# this change to _to_int should at least prevent mpl from crashing on
# these JDH (2009-11-06)

def _to_int(x):
    return int(float(x))

_to_float = float
_to_str =  str

def _to_list_of_ints(s):
    s = s.replace(',', ' ')
    return [_to_int(val) for val in s.split()]

def _to_list_of_floats(s):
    return [_to_float(val) for val in s.split()]
def _to_bool(s):
    if s.lower().strip() in ('false', '0', 'no'): return False
    else: return True


def _sanity_check(fh):
    """
    Check if the file at least looks like AFM.
    If not, raise :exc:`RuntimeError`.
    """

    # Remember the file position in case the caller wants to
    # do something else with the file.
    pos = fh.tell()
    try:
        line = fh.readline()
    finally:
        fh.seek(pos, 0)

    # AFM spec, Section 4: The StartFontMetrics keyword [followed by a
    # version number] must be the first line in the file, and the
    # EndFontMetrics keyword must be the last non-empty line in the
    # file. We just check the first line.
    if not line.startswith('StartFontMetrics'):
        raise RuntimeError('Not an AFM file')

def _parse_header(fh):
    """
    Reads the font metrics header (up to the char metrics) and returns
    a dictionary mapping *key* to *val*.  *val* will be converted to the
    appropriate python type as necessary; eg:

        * 'False'->False
        * '0'->0
        * '-168 -218 1000 898'-> [-168, -218, 1000, 898]

    Dictionary keys are

      StartFontMetrics, FontName, FullName, FamilyName, Weight,
      ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,
      UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,
      XHeight, Ascender, Descender, StartCharMetrics

    """
    headerConverters = {
        'StartFontMetrics': _to_float,
        'FontName': _to_str,
        'FullName': _to_str,
        'FamilyName': _to_str,
        'Weight': _to_str,
        'ItalicAngle': _to_float,
        'IsFixedPitch': _to_bool,
        'FontBBox': _to_list_of_ints,
        'UnderlinePosition': _to_int,
        'UnderlineThickness': _to_int,
        'Version': _to_str,
        'Notice': _to_str,
        'EncodingScheme': _to_str,
        'CapHeight': _to_float, # Is the second version a mistake, or
        'Capheight': _to_float, # do some AFM files contain 'Capheight'? -JKS
        'XHeight': _to_float,
        'Ascender': _to_float,
        'Descender': _to_float,
        'StdHW': _to_float,
        'StdVW': _to_float,
        'StartCharMetrics': _to_int,
        'CharacterSet': _to_str,
        'Characters': _to_int,
        }

    d = {}
    while 1:
        line = fh.readline()
        if not line: break
        line = line.rstrip()
        if line.startswith('Comment'): continue
        lst = line.split( ' ', 1 )
        #print '%-s\t%-d line :: %-s' % ( fh.name, len(lst), line )
        key = lst[0]
        if len( lst ) == 2:
            val = lst[1]
        else:
            val = ''
        #key, val = line.split(' ', 1)
        try: d[key] = headerConverters[key](val)
        except ValueError:
            print >>sys.stderr, 'Value error parsing header in AFM:', key, val
            continue
        except KeyError:
            print >>sys.stderr, 'Found an unknown keyword in AFM header (was %s)' % key
            continue
        if key=='StartCharMetrics': return d
    raise RuntimeError('Bad parse')

def _parse_char_metrics(fh):
    """
    Return a character metric dictionary.  Keys are the ASCII num of
    the character, values are a (*wx*, *name*, *bbox*) tuple, where
    *wx* is the character width, *name* is the postscript language
    name, and *bbox* is a (*llx*, *lly*, *urx*, *ury*) tuple.

    This function is incomplete per the standard, but thus far parses
    all the sample afm files tried.
    """

    ascii_d = {}
    name_d  = {}
    while 1:
        line = fh.readline()
        if not line: break
        line = line.rstrip()
        if line.startswith('EndCharMetrics'): return ascii_d, name_d
        vals = line.split(';')[:4]
        if len(vals) !=4 : raise RuntimeError('Bad char metrics line: %s' % line)
        num = _to_int(vals[0].split()[1])
        wx = _to_float(vals[1].split()[1])
        name = vals[2].split()[1]
        bbox = _to_list_of_floats(vals[3][2:])
        bbox = map(int, bbox)
        # Workaround: If the character name is 'Euro', give it the corresponding
        # character code, according to WinAnsiEncoding (see PDF Reference).
        if name == 'Euro':
            num = 128
        if num != -1:
            ascii_d[num] = (wx, name, bbox)
        name_d[name] = (wx, bbox)
    raise RuntimeError('Bad parse')

def _parse_kern_pairs(fh):
    """
    Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and
    values are the kern pair value.  For example, a kern pairs line like
    ``KPX A y -50``

    will be represented as::

      d[ ('A', 'y') ] = -50

    """

    line = fh.readline()
    if not line.startswith('StartKernPairs'):
        raise RuntimeError('Bad start of kern pairs data: %s'%line)

    d = {}
    while 1:
        line = fh.readline()
        if not line: break
        line = line.rstrip()
        if len(line)==0: continue
        if line.startswith('EndKernPairs'):
            fh.readline()  # EndKernData
            return d
        vals = line.split()
        if len(vals)!=4 or vals[0]!='KPX':
            raise RuntimeError('Bad kern pairs line: %s'%line)
        c1, c2, val = vals[1], vals[2], _to_float(vals[3])
        d[(c1,c2)] = val
    raise RuntimeError('Bad kern pairs parse')

def _parse_composites(fh):
    """
    Return a composites dictionary.  Keys are the names of the
    composites.  Values are a num parts list of composite information,
    with each element being a (*name*, *dx*, *dy*) tuple.  Thus a
    composites line reading:

      CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;

    will be represented as::

      d['Aacute'] = [ ('A', 0, 0), ('acute', 160, 170) ]

    """
    d = {}
    while 1:
        line = fh.readline()
        if not line: break
        line = line.rstrip()
        if len(line)==0: continue
        if line.startswith('EndComposites'):
            return d
        vals = line.split(';')
        cc = vals[0].split()
        name, numParts = cc[1], _to_int(cc[2])
        pccParts = []
        for s in vals[1:-1]:
            pcc = s.split()
            name, dx, dy = pcc[1], _to_float(pcc[2]), _to_float(pcc[3])
            pccParts.append( (name, dx, dy) )
        d[name] = pccParts

    raise RuntimeError('Bad composites parse')

def _parse_optional(fh):
    """
    Parse the optional fields for kern pair data and composites

    return value is a (*kernDict*, *compositeDict*) which are the
    return values from :func:`_parse_kern_pairs`, and
    :func:`_parse_composites` if the data exists, or empty dicts
    otherwise
    """
    optional = {
        'StartKernData' : _parse_kern_pairs,
        'StartComposites' :  _parse_composites,
        }

    d = {'StartKernData':{}, 'StartComposites':{}}
    while 1:
        line = fh.readline()
        if not line: break
        line = line.rstrip()
        if len(line)==0: continue
        key = line.split()[0]

        if key in optional: d[key] = optional[key](fh)

    l = ( d['StartKernData'], d['StartComposites'] )
    return l

def parse_afm(fh):
    """
    Parse the Adobe Font Metics file in file handle *fh*. Return value
    is a (*dhead*, *dcmetrics*, *dkernpairs*, *dcomposite*) tuple where
    *dhead* is a :func:`_parse_header` dict, *dcmetrics* is a
    :func:`_parse_composites` dict, *dkernpairs* is a
    :func:`_parse_kern_pairs` dict (possibly {}), and *dcomposite* is a
    :func:`_parse_composites` dict (possibly {})
    """
    _sanity_check(fh)
    dhead =  _parse_header(fh)
    dcmetrics_ascii, dcmetrics_name = _parse_char_metrics(fh)
    doptional = _parse_optional(fh)
    return dhead, dcmetrics_ascii, dcmetrics_name, doptional[0], doptional[1]


class AFM:

    def __init__(self, fh):
        """
        Parse the AFM file in file object *fh*
        """
        (dhead, dcmetrics_ascii, dcmetrics_name, dkernpairs, dcomposite) = \
            parse_afm(fh)
        self._header = dhead
        self._kern = dkernpairs
        self._metrics = dcmetrics_ascii
        self._metrics_by_name = dcmetrics_name
        self._composite = dcomposite


    def get_bbox_char(self, c, isord=False):
        if not isord: c=ord(c)
        wx, name, bbox = self._metrics[c]
        return bbox

    def string_width_height(self, s):
        """
        Return the string width (including kerning) and string height
        as a (*w*, *h*) tuple.
        """
        if not len(s): return 0,0
        totalw = 0
        namelast = None
        miny = 1e9
        maxy = 0
        for c in s:
            if c == '\n': continue
            wx, name, bbox = self._metrics[ord(c)]
            l,b,w,h = bbox

            # find the width with kerning
            try: kp = self._kern[ (namelast, name) ]
            except KeyError: kp = 0
            totalw += wx + kp

            # find the max y
            thismax = b+h
            if thismax>maxy: maxy = thismax

            # find the min y
            thismin = b
            if thismin<miny: miny = thismin

        return totalw, maxy-miny

    def get_str_bbox_and_descent(self, s):
        """
        Return the string bounding box
        """
        if not len(s): return 0,0,0,0
        totalw = 0
        namelast = None
        miny = 1e9
        maxy = 0
        left = 0
        if not isinstance(s, unicode):
            s = s.decode()
        for c in s:
            if c == '\n': continue
            name = uni2type1.get(ord(c), 'question')
            try:
                wx, bbox = self._metrics_by_name[name]
            except KeyError:
                name = 'question'
                wx, bbox = self._metrics_by_name[name]
            l,b,w,h = bbox
            if l<left: left = l
            # find the width with kerning
            try: kp = self._kern[ (namelast, name) ]
            except KeyError: kp = 0
            totalw += wx + kp

            # find the max y
            thismax = b+h
            if thismax>maxy: maxy = thismax

            # find the min y
            thismin = b
            if thismin<miny: miny = thismin

        return left, miny, totalw, maxy-miny, -miny


    def get_str_bbox(self, s):
        """
        Return the string bounding box
        """
        return self.get_str_bbox_and_descent(s)[:4]

    def get_name_char(self, c, isord=False):
        """
        Get the name of the character, ie, ';' is 'semicolon'
        """
        if not isord: c=ord(c)
        wx, name, bbox = self._metrics[c]
        return name

    def get_width_char(self, c, isord=False):
        """
        Get the width of the character from the character metric WX
        field
        """
        if not isord: c=ord(c)
        wx, name, bbox = self._metrics[c]
        return wx

    def get_width_from_char_name(self, name):
        """
        Get the width of the character from a type1 character name
        """
        wx, bbox = self._metrics_by_name[name]
        return wx

    def get_height_char(self, c, isord=False):
        """
        Get the height of character *c* from the bounding box.  This
        is the ink height (space is 0)
        """
        if not isord: c=ord(c)
        wx, name, bbox = self._metrics[c]
        return bbox[-1]

    def get_kern_dist(self, c1, c2):
        """
        Return the kerning pair distance (possibly 0) for chars *c1*
        and *c2*
        """
        name1, name2 = self.get_name_char(c1), self.get_name_char(c2)
        return self.get_kern_dist_from_name(name1, name2)

    def get_kern_dist_from_name(self, name1, name2):
        """
        Return the kerning pair distance (possibly 0) for chars
        *name1* and *name2*
        """
        try: return self._kern[ (name1, name2) ]
        except: return 0

    def get_fontname(self):
        "Return the font name, eg, 'Times-Roman'"
        return self._header['FontName']

    def get_fullname(self):
        "Return the font full name, eg, 'Times-Roman'"
        name = self._header.get('FullName')
        if name is None: # use FontName as a substitute
            name = self._header['FontName']
        return name

    def get_familyname(self):
        "Return the font family name, eg, 'Times'"
        name = self._header.get('FamilyName')
        if name is not None:
            return name

        # FamilyName not specified so we'll make a guess
        name = self.get_fullname()
        extras = r'(?i)([ -](regular|plain|italic|oblique|bold|semibold|light|ultralight|extra|condensed))+$'
        return re.sub(extras, '', name)

    def get_weight(self):
        "Return the font weight, eg, 'Bold' or 'Roman'"
        return self._header['Weight']

    def get_angle(self):
        "Return the fontangle as float"
        return self._header['ItalicAngle']

    def get_capheight(self):
        "Return the cap height as float"
        return self._header['CapHeight']

    def get_xheight(self):
        "Return the xheight as float"
        return self._header['XHeight']

    def get_underline_thickness(self):
        "Return the underline thickness as float"
        return self._header['UnderlineThickness']

    def get_horizontal_stem_width(self):
        """
        Return the standard horizontal stem width as float, or *None* if
        not specified in AFM file.
        """
        return self._header.get('StdHW', None)

    def get_vertical_stem_width(self):
        """
        Return the standard vertical stem width as float, or *None* if
        not specified in AFM file.
        """
        return self._header.get('StdVW', None)


if __name__=='__main__':
    #pathname = '/usr/local/lib/R/afm/'
    pathname = '/usr/local/share/fonts/afms/adobe'

    for fname in os.listdir(pathname):
        fh = file(os.path.join(pathname,fname))
        afm = AFM(fh)
        w,h =  afm.string_width_height('John Hunter is the Man!')
