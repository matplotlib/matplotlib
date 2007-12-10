"""
A module for finding, managing, and using fonts across-platforms.

This module provides a single FontManager that can be shared across
backends and platforms.  The findfont() function returns the best
TrueType (TTF) font file in the local or system font path that matches
the specified FontProperties.  The FontManager also handles Adobe Font
Metrics (AFM) font files for use by the PostScript backend.

The design is based on the W3C Cascading Style Sheet, Level 1 (CSS1)
font specification (http://www.w3.org/TR/1998/REC-CSS2-19980512/ ).
Future versions may implement the Level 2 or 2.1 specifications.


KNOWN ISSUES

  - documentation
  - font variant is untested
  - font stretch is incomplete
  - font size is incomplete
  - font size_adjust is incomplete
  - default font algorithm needs improvement and testing
  - setWeights function needs improvement
  - 'light' is an invalid weight value, remove it.
  - update_fonts not implemented


Authors   : John Hunter <jdhunter@ace.bsd.uchicago.edu>
            Paul Barrett <Barrett@STScI.Edu>
Copyright : John Hunter (2004,2005), Paul Barrett (2004,2005)
License   : matplotlib license (PSF compatible)
            The font directory code is from ttfquery,
            see license/LICENSE_TTFQUERY.
"""

import os, sys, glob, shutil
from sets import Set
import matplotlib
from matplotlib import afm
from matplotlib import ft2font
from matplotlib import rcParams, get_home, get_configdir
from matplotlib.cbook import is_string_like
from matplotlib.fontconfig_pattern import \
    parse_fontconfig_pattern, generate_fontconfig_pattern

try:
    import cPickle as pickle
except ImportError:
    import pickle

USE_FONTCONFIG = False

verbose = matplotlib.verbose

font_scalings = {'xx-small': 0.579, 'x-small': 0.694, 'small': 0.833,
                 'medium': 1.0, 'large': 1.200, 'x-large': 1.440,
                 'xx-large': 1.728, 'larger': 1.2, 'smaller': 0.833}

weight_dict = {'light': 200, 'normal': 400, 'regular': 400, 'book': 400,
               'medium': 500, 'roman': 500, 'semibold': 600, 'demibold': 600,
               'demi': 600, 'bold': 700, 'heavy': 800, 'extra bold': 800,
               'black': 900}

#  OS Font paths
MSFolders = \
    r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'

MSFontDirectories   = [
    r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts',
    r'SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts']

X11FontDirectories  = [
    # an old standard installation point
    "/usr/X11R6/lib/X11/fonts/TTF/",
    # here is the new standard location for fonts
    "/usr/share/fonts/",
    # documented as a good place to install new fonts
    "/usr/local/share/fonts/",
    # common application, not really useful
    "/usr/lib/openoffice/share/fonts/truetype/",
    ]

OSXFontDirectories = [
    "/Library/Fonts/",
    "/Network/Library/Fonts/",
    "/System/Library/Fonts/"
]

if not USE_FONTCONFIG:
    home = os.environ.get('HOME')
    if home is not None:
        # user fonts on OSX
        path = os.path.join(home, 'Library', 'Fonts')
        OSXFontDirectories.append(path)
        path = os.path.join(home, '.fonts')
        X11FontDirectories.append(path)

def get_fontext_synonyms(fontext):
    return {'ttf': ('ttf', 'otf'),
            'afm': ('afm',)}[fontext]

def win32FontDirectory():
    """Return the user-specified font directory for Win32."""

    try:
        import _winreg
    except ImportError:
        pass # Fall through to default
    else:
        user = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, MSFolders)
        try:
            try:
                return _winreg.QueryValueEx(user, 'Fonts')[0]
            except OSError:
                pass # Fall through to default
        finally:
            _winreg.CloseKey(user)
    return os.path.join(os.environ['WINDIR'], 'Fonts')

def win32InstalledFonts(directory=None, fontext='ttf'):
    """
    Search for fonts in the specified font directory, or use the
    system directories if none given.  A list of TrueType fonts are
    returned by default with AFM fonts as an option.
    """

    import _winreg
    if directory is None:
        directory = win32FontDirectory()

    fontext = get_fontext_synonyms(fontext)

    key, items = None, {}
    for fontdir in MSFontDirectories:
        try:
            local = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, fontdir)
        except OSError:
            continue

        if not local:
            files = []
            for ext in fontext:
                files.extend(glob.glob(os.path.join(directory, '*.'+ext)))
            return files
        try:
            for j in range(_winreg.QueryInfoKey(local)[1]):
                try:
                    key, direc, any = _winreg.EnumValue( local, j)
                    if not os.path.dirname(direc):
                        direc = os.path.join(directory, direc)
                    direc = os.path.abspath(direc).lower()
                    if os.path.splitext(direc)[1][1:] in fontext:
                        items[direc] = 1
                except EnvironmentError:
                    continue
                except WindowsError:
                    continue

            return items.keys()
        finally:
            _winreg.CloseKey(local)
    return None

def OSXFontDirectory():
    """Return the system font directories for OS X."""

    fontpaths = []
    def add(arg,directory,files):
        fontpaths.append(directory)
    for fontdir in OSXFontDirectories:
        try:
            if os.path.isdir(fontdir):
                os.path.walk(fontdir, add, None)
        except (IOError, OSError, TypeError, ValueError):
            pass
    return fontpaths

def OSXInstalledFonts(directory=None, fontext='ttf'):
    """Get list of font files on OS X - ignores font suffix by default"""
    if directory is None:
        directory = OSXFontDirectory()

    fontext = get_fontext_synonyms(fontext)

    files = []
    for path in directory:
        if fontext is None:
            files.extend(glob.glob(os.path.join(path,'*')))
        else:
            for ext in fontext:
                files.extend(glob.glob(os.path.join(path, '*.'+ext)))
                files.extend(glob.glob(os.path.join(path, '*.'+ext.upper())))
    return files


def x11FontDirectory():
    """Return the system font directories for X11."""
    fontpaths = []
    def add(arg,directory,files):
        fontpaths.append(directory)

    for fontdir in X11FontDirectories:
        try:
            if os.path.isdir(fontdir):
                os.path.walk(fontdir, add, None)
        except (IOError, OSError, TypeError, ValueError):
            pass
    return fontpaths

def get_fontconfig_fonts(fontext='ttf'):
    """Grab a list of all the fonts that are being tracked by fontconfig.
    This is an easy way to grab all of the fonts the user wants to be made
    available to applications, without knowing where all of them reside."""
    try:
        import commands
    except ImportError:
        return {}

    fontext = get_fontext_synonyms(fontext)

    fontfiles = {}
    status, output = commands.getstatusoutput("fc-list file")
    if status == 0:
        for line in output.split('\n'):
            fname = line.split(':')[0]
            if (os.path.splitext(fname)[1][1:] in fontext and
                os.path.exists(fname)):
                fontfiles[fname] = 1

    return fontfiles

def findSystemFonts(fontpaths=None, fontext='ttf'):
    """
    Search for fonts in the specified font paths.  If no paths are
    given, will use a standard set of system paths, as well as the
    list of fonts tracked by fontconfig if fontconfig is installed and
    available.  A list of TrueType fonts are returned by default with
    AFM fonts as an option.
    """
    fontfiles = {}
    fontexts = get_fontext_synonyms(fontext)

    if fontpaths is None:
        if sys.platform == 'win32':
            fontdir = win32FontDirectory()

            fontpaths = [fontdir]
            # now get all installed fonts directly...
            for f in win32InstalledFonts(fontdir):
                base, ext = os.path.splitext(f)
                if len(ext)>1 and ext[1:].lower() in fontexts:
                    fontfiles[f] = 1
        else:
            fontpaths = x11FontDirectory()
            # check for OS X & load its fonts if present
            if sys.platform == 'darwin':
                for f in OSXInstalledFonts(fontext=fontext):
                    fontfiles[f] = 1

            for f in get_fontconfig_fonts(fontext):
                fontfiles[f] = 1

    elif isinstance(fontpaths, (str, unicode)):
        fontpaths = [fontpaths]

    for path in fontpaths:
        files = []
        for ext in fontexts:
            files.extend(glob.glob(os.path.join(path, '*.'+ext)))
            files.extend(glob.glob(os.path.join(path, '*.'+ext.upper())))
        for fname in files:
            fontfiles[os.path.abspath(fname)] = 1

    return [fname for fname in fontfiles.keys() if os.path.exists(fname)]

def weight_as_number(weight):
    """
    Return the weight property as a numeric value.  String values
    are converted to their corresponding numeric value.
    """
    if isinstance(weight, str):
        try:
            weight = weight_dict[weight.lower()]
        except KeyError:
            weight = 400
    elif weight in range(100, 1000, 100):
        pass
    else:
        raise ValueError, 'weight not a valid integer'
    return weight


class FontKey(object):
    """
    A class for storing Font properties.  It is used when populating
    the font dictionary.
    """

    def __init__(self,
                 name   ='',
                 style  ='normal',
                 variant='normal',
                 weight ='normal',
                 stretch='normal',
                 size   ='medium'
                 ):
        self.name    = name
        self.style   = style
        self.variant = variant
        self.weight  = weight
        self.stretch = stretch
        try:
            self.size = str(float(size))
        except ValueError:
            self.size = size


def ttfFontProperty(font):
    """
    A function for populating the FontKey by extracting information
    from the TrueType font file.
    """
    name = font.family_name

    #  Styles are: italic, oblique, and normal (default)

    sfnt = font.get_sfnt()
    sfnt2 = sfnt.get((1,0,0,2))
    sfnt4 = sfnt.get((1,0,0,4))
    if sfnt2:
        sfnt2 = sfnt2.lower()
    else:
        sfnt2 = ''
    if sfnt4:
        sfnt4 = sfnt4.lower()
    else:
        sfnt4 = ''
    if   sfnt4.find('oblique') >= 0:
        style = 'oblique'
    elif sfnt4.find('italic') >= 0:
        style = 'italic'
    elif sfnt2.find('regular') >= 0:
        style = 'normal'
    elif font.style_flags & ft2font.ITALIC:
        style = 'italic'
    else:
        style = 'normal'


    #  Variants are: small-caps and normal (default)

    #  !!!!  Untested
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.

    weight = None
    for w in weight_dict.keys():
        if sfnt4.find(w) >= 0:
            weight = w
            break
    if not weight:
        if font.style_flags & ft2font.BOLD:
            weight = 700
        else:
            weight = 400
    weight = weight_as_number(weight)

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit

    #  !!!!  Incomplete
    if   sfnt4.find('narrow') >= 0 or sfnt4.find('condensed') >= 0 or \
           sfnt4.find('cond') >= 0:
        stretch = 'condensed'
    elif sfnt4.find('demi cond') >= 0:
        stretch = 'semi-condensed'
    elif sfnt4.find('wide') >= 0 or sfnt4.find('expanded') >= 0:
        stretch = 'expanded'
    else:
        stretch = 'normal'

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g. 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    #  !!!!  Incomplete
    if font.scalable:
        size = 'scalable'
    else:
        size = str(float(font.get_fontsize()))

    #  !!!!  Incomplete
    size_adjust = None

    return FontKey(name, style, variant, weight, stretch, size)


def afmFontProperty(font):
    """
    A function for populating the FontKey by extracting information
    from the AFM font file.
    """

    name = font.get_familyname()

    #  Styles are: italic, oblique, and normal (default)

    if font.get_angle() != 0 or name.lower().find('italic') >= 0:
        style = 'italic'
    elif name.lower().find('oblique') >= 0:
        style = 'oblique'
    else:
        style = 'normal'

    #  Variants are: small-caps and normal (default)

    # !!!!  Untested
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.

    weight = weight_as_number(font.get_weight().lower())

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit

    # !!!!  Incomplete
    stretch = 'normal'

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g. 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    #  All AFM fonts are apparently scalable.

    size = 'scalable'

    # !!!!  Incomplete
    size_adjust = None

    return FontKey(name, style, variant, weight, stretch, size)


def add_filename(fontdict, prop, fname):
    """
    A function to add a font file name to the font dictionary using
    the FontKey properties.  If a font property has no dictionary, then
    create it.
    """
    try:
        size = str(float(prop.size))
    except ValueError:
        size = prop.size

    d = fontdict.                    \
        setdefault(prop.name,    {}).\
        setdefault(prop.style,   {}).\
        setdefault(prop.variant, {}).\
        setdefault(prop.weight,  {}).\
        setdefault(prop.stretch, {})
    d[size] = fname


def createFontDict(fontfiles, fontext='ttf'):
    """
    A function to create a dictionary of font file paths.  The
    default is to create a dictionary for TrueType fonts.  An AFM font
    dictionary can optionally be created.
    """

    fontdict = {}
    #  Add fonts from list of known font files.
    seen = {}
    for fpath in fontfiles:
        verbose.report('createFontDict: %s' % (fpath), 'debug')
        fname = os.path.split(fpath)[1]
        if seen.has_key(fname):  continue
        else: seen[fname] = 1
        if fontext == 'afm':
            try:
                fh = open(fpath, 'r')
            except:
                verbose.report("Could not open font file %s" % fpath)
                continue
            try:
                try:
                    font = afm.AFM(fh)
                finally:
                    fh.close()
            except RuntimeError:
                verbose.report("Could not parse font file %s"%fpath)
                continue
            prop = afmFontProperty(font)
        else:
            try:
                font = ft2font.FT2Font(str(fpath))
            except RuntimeError:
                verbose.report("Could not open font file %s"%fpath)
                continue
            except UnicodeError:
                verbose.report("Cannot handle unicode filenames")
                #print >> sys.stderr, 'Bad file is', fpath
                continue
            try: prop = ttfFontProperty(font)
            except: continue

        add_filename(fontdict, prop, fpath)
    return fontdict

def setWeights(font):
    """
    A function to populate missing values in a font weight
    dictionary.  This proceedure is necessary since the font finding
    algorithm always matches on the weight property.
    """

    # !!!!  Not completely correct
    temp = font.copy()
    if len(temp) == 1:
        wgt = temp.keys()[0]
        for j in range(100, 1000, 100):
            font[j] = temp[wgt]

    if temp.has_key(400):
        for j in range(100, 1000, 100):
            font[j] = temp[400]
    if temp.has_key(500):
        if temp.has_key(400):
            for j in range(500, 1000, 100):
                font[j] = temp[500]
        else:
            for j in range(100, 1000, 100):
                font[j] = temp[500]

    if temp.has_key(300):
        for j in [100, 200, 300]:
            font[j] = temp[300]
    if temp.has_key(200):
        if temp.has_key(300):
            for j in [100, 200]:
                font[j] = temp[200]
        else:
            for j in [100, 200, 300]:
                font[j] = temp[200]

    if temp.has_key(800):
        for j in [600, 700, 800, 900]:
            font[j] = temp[800]
    if temp.has_key(700):
        if temp.has_key(800):
            for j in [600, 700]:
                font[j] = temp[700]
        else:
            for j in [600, 700, 800, 900]:
                font[j] = temp[700]

class FontProperties(object):
    """
    A class for storing and manipulating font properties.

    The font properties are those described in the W3C Cascading Style
    Sheet, Level 1 (CSS1; http://www.w3.org/TR/1998/REC-CSS2-19980512/)
    font specification.  The six properties are:

      family  - A list of font names in decreasing order of priority.
                The last item is the default font name and is given the
                name of the font family, either serif, sans-serif,
                cursive, fantasy, and monospace.
      style   - Either normal, italic or oblique.
      variant - Either normal or small-caps.
      stretch - Either an absolute value of ultra-condensed, extra-
                condensed, condensed, semi-condensed, normal, semi-
                expanded, expanded, extra-expanded or ultra-expanded;
                or a relative value of narrower or wider.
                This property is currently not implemented and is set to
                normal.
      weight  - A numeric value in the range 100, 200, 300, ..., 900.
      size    - Either an absolute value of xx-small, x-small, small,
                medium, large, x-large, xx-large; or a relative value
                of smaller or larger; or an absolute font size, e.g. 12;
                or scalable.

    The default font property for TrueType fonts is: sans-serif, normal,
    normal, normal, 400, scalable.

    The preferred usage of font sizes is to use the relative values, e.g.
    large, instead of absolute font sizes, e.g. 12.  This approach allows
    all text sizes to be made larger or smaller based on the font manager's
    default font size, i.e. by using the set_default_size() method of the
    font manager.

    This class will also accept a fontconfig pattern, if it is the only
    argument provided.  fontconfig patterns are described here:

      http://www.fontconfig.org/fontconfig-user.html

    Note that matplotlib's internal font manager and fontconfig use a
    different algorithm to lookup fonts, so the results of the same pattern
    may be different in matplotlib than in other applications that use
    fontconfig.
    """

    class FontPropertiesSet(object):
        """This class contains all of the default properties at the
        class level, which are then overridden (only if provided) at
        the instance level."""
        family = rcParams['font.' + rcParams['font.family']]
        if is_string_like(family):
            family = [family]
        slant = [rcParams['font.style']]
        variant = [rcParams['font.variant']]
        weight = [rcParams['font.weight']]
        stretch = [rcParams['font.stretch']]
        size = [rcParams['font.size']]
        file = None

    def __init__(self,
                 family = None,
                 style  = None,
                 variant= None,
                 weight = None,
                 stretch= None,
                 size   = None,
                 fname  = None, # if this is set, it's a hardcoded filename to use
                 _init   = None  # used only by copy()
                 ):

        self.__props = self.FontPropertiesSet()

        # This is used only by copy()
        if _init is not None:
            self.__props.__dict__.update(_init)
            return

        if is_string_like(family):
            # Treat family as a fontconfig pattern if it is the only
            # parameter provided.
            if (style is None and
                variant is None and
                weight is None and
                stretch is None and
                size is None and
                fname is None):
                self.__props.__dict__ = self._parse_fontconfig_pattern(family)
                return
            family = [family]

        self.set_family(family)
        self.set_style(style)
        self.set_variant(variant)
        self.set_weight(weight)
        self.set_stretch(stretch)
        self.set_file(fname)
        self.set_size(size)

    def _parse_fontconfig_pattern(self, pattern):
        return parse_fontconfig_pattern(pattern)

    def __hash__(self):
        return hash(repr(self.__props.__dict__))

    def __str__(self):
        return self.get_fontconfig_pattern()

    def get_family(self):
        """Return a list of font names that comprise the font family.
        """
        return self.__props.family

    def get_name(self):
        """Return the name of the font that best matches the font properties."""
        return ft2font.FT2Font(str(findfont(self))).family_name

    def get_style(self):
        """Return the font style.  Values are: normal, italic or oblique."""
        return self.__props.slant[0]

    def get_variant(self):
        """Return the font variant.  Values are: normal or small-caps."""
        return self.__props.variant[0]

    def get_weight(self):
        """
        Return the font weight.  See the FontProperties class for a
        a list of possible values.
        """
        return self.__props.weight[0]

    def get_stretch(self):
        """
        Return the font stretch or width.  Options are: normal,
        narrow, condensed, or wide.
        """
        return self.__props.stretch[0]

    def get_size(self):
        """Return the font size."""
        return float(self.__props.size[0])

    def get_file(self):
        if self.__props.file is not None:
            return self.__props.file[0]
        else:
            return None

    def get_fontconfig_pattern(self):
        return generate_fontconfig_pattern(self.__props.__dict__)

    def set_family(self, family):
        """
        Change the font family.  May be either an alias (generic name
        is CSS parlance), such as: serif, sans-serif, cursive,
        fantasy, or monospace, or a real font name.
        """
        if family is None:
            self.__props.__dict__.pop('family', None)
        else:
            if is_string_like(family):
                family = [family]
            self.__props.family = family
    set_name = set_family

    def set_style(self, style):
        """Set the font style.  Values are: normal, italic or oblique."""
        if style is None:
            self.__props.__dict__.pop('style', None)
        else:
            if style not in ('normal', 'italic', 'oblique'):
                raise ValueError("style must be normal, italic or oblique")
            self.__props.slant = [style]

    def set_variant(self, variant):
        """Set the font variant.  Values are: normal or small-caps."""
        if variant is None:
            self.__props.__dict__.pop('variant', None)
        else:
            if variant not in ('normal', 'small-caps'):
                raise ValueError("variant must be normal or small-caps")
            self.__props.variant = [variant]

    def set_weight(self, weight):
        """
        Set the font weight.  See the FontProperties class for a
        a list of possible values.
        """
        if weight is None:
            self.__props.__dict__.pop('weight', None)
        else:
            if (weight not in weight_dict and
                weight not in weight_dict.keys()):
                raise ValueError("weight is invalid")
            self.__props.weight = [weight]

    def set_stretch(self, stretch):
        """
        Set the font stretch or width.  Options are: normal, narrow,
        condensed, or wide.
        """
        if stretch is None:
            self.__props.__dict__.pop('stretch', None)
        else:
            self.__props.stretch = [stretch]

    def set_size(self, size):
        """Set the font size."""
        if size is None:
            self.__props.__dict__.pop('size', None)
        else:
            if is_string_like(size):
                parent_size = fontManager.get_default_size()
                scaling = font_scalings.get(size)
                if scaling is not None:
                    size = parent_size * scaling
                else:
                    size = parent_size
            if isinstance(size, (int, float)):
                size = [size]
            self.__props.size = size

    def set_file(self, file):
        if file is None:
            self.__props.__dict__.pop('file', None)
        else:
            self.__props.file = [file]

    get_size_in_points = get_size

    def set_fontconfig_pattern(self, pattern):
        self.__props.__dict__ = self._parse_fontconfig_pattern(pattern)

    def add_property_pair(self, key, val):
        self.__props.setdefault(key, []).append(val)

    def copy(self):
        """Return a deep copy of self"""
        return FontProperties(_init = self.__props.__dict__)

def ttfdict_to_fnames(d):
    'flatten a ttfdict to all the filenames it contains'
    fnames = []
    for named in d.values():
        for styled in named.values():
            for variantd in styled.values():
                for weightd in variantd.values():
                    for stretchd in weightd.values():
                        for fname in stretchd.values():
                            fnames.append(fname)
    return fnames

def pickle_dump(data, filename):
    """Equivalent to pickle.dump(data, open(filename, 'w'))
    but closes the file to prevent filehandle leakage."""
    fh = open(filename, 'w')
    try:
        pickle.dump(data, fh)
    finally:
        fh.close()

def pickle_load(filename):
    """Equivalent to pickle.load(open(filename, 'r'))
    but closes the file to prevent filehandle leakage."""
    fh = open(filename, 'r')
    try:
        data = pickle.load(fh)
    finally:
        fh.close()
    return data

class FontManager:
    """
    On import, the FontManager creates a dictionary of TrueType
    fonts based on the font properties: name, style, variant, weight,
    stretch, and size.  The findfont() method searches this dictionary
    for a font file name that exactly matches the font properties of the
    specified text.  If none is found, a default font is returned.  By
    updating the dictionary with the properties of the found font, the
    font dictionary can act like a font cache.
    """

    def __init__(self, size=None, weight='normal'):
        self.__default_weight = weight
        self.default_size = size

        paths = [os.path.join(rcParams['datapath'],'fonts','ttf'),
                 os.path.join(rcParams['datapath'],'fonts','afm')]

        #  Create list of font paths

        for pathname in ['TTFPATH', 'AFMPATH']:
            if os.environ.has_key(pathname):
                ttfpath = os.environ[pathname]
                if ttfpath.find(';') >= 0: #win32 style
                    paths.extend(ttfpath.split(';'))
                elif ttfpath.find(':') >= 0: # unix style
                    paths.extend(ttfpath.split(':'))
                else:
                    paths.append(ttfpath)

        verbose.report('font search path %s'%(str(paths)))
        #  Load TrueType fonts and create font dictionary.

        self.ttffiles = findSystemFonts(paths) + findSystemFonts()

        for fname in self.ttffiles:
            verbose.report('trying fontname %s' % fname, 'debug')
            if fname.lower().find('vera.ttf')>=0:
                self.defaultFont = fname
                break
        else:
            # use anything
            self.defaultFont = self.ttffiles[0]

        self.ttfdict = createFontDict(self.ttffiles)

        if rcParams['pdf.use14corefonts']:
            # Load only the 14 PDF core fonts. These fonts do not need to be
            # embedded; every PDF viewing application is required to have them:
            # Helvetica, Helvetica-Bold, Helvetica-Oblique, Helvetica-BoldOblique,
            # Courier, Courier-Bold, Courier-Oblique, Courier-BoldOblique,
            # Times-Roman, Times-Bold, Times-Italic, Times-BoldItalic, Symbol,
            # ZapfDingbats.
            afmpath = os.path.join(rcParams['datapath'],'fonts','pdfcorefonts')
            afmfiles = findSystemFonts(afmpath, fontext='afm')
            self.afmdict = createFontDict(afmfiles, fontext='afm')
        else:
            self.afmfiles = findSystemFonts(paths, fontext='afm') + \
                            findSystemFonts(fontext='afm')
            self.afmdict = createFontDict(self.afmfiles, fontext='afm')

    def get_default_weight(self):
        "Return the default font weight."
        return self.__default_weight

    def get_default_size(self):
        "Return the default font size."
        if self.default_size is None:
            return rcParams['font.size']
        return self.default_size

    def set_default_weight(self, weight):
        "Set the default font weight.  The initial value is 'normal'."
        self.__default_weight = weight

    def set_default_size(self, size):
        "Set the default font size in points.  The initial value is set by font.size in rc."
        self.__default_size = size

    def update_fonts(self, filenames):
        """
        Update the font dictionary with new font files.
        Currently not implemented.
        """
        #  !!!!  Needs implementing
        raise NotImplementedError

    def findfont(self, prop, fontext='ttf'):
        """
        Search the font dictionary for a font that exactly or closely
        matches the specified font properties.  See the FontProperties class
        for a description.

        The properties are searched in the following order: name, style,
        variant, weight, stretch, and size.  The font weight always matches
        returning the closest weight, and the font size always matches for
        scalable fonts.  An oblique style font will be used inplace of a
        missing italic style font if present.  See the W3C Cascading Style
        Sheet, Level 1 (CSS1; http://www.w3.org/TR/1998/REC-CSS2-19980512/)
        documentation for a description of the font finding algorithm.
        """
        debug = False
        if is_string_like(prop):
            prop = FontProperties(prop)
        fname = prop.get_file()
        if fname is not None:
            verbose.report('findfont returning %s'%fname, 'debug')
            return fname

        if fontext == 'afm':
            fontdict = self.afmdict
        else:
            fontdict = self.ttfdict

        original_name = prop.get_family()[0]
        style         = prop.get_style()
        variant       = prop.get_variant()
        weight        = weight_as_number(prop.get_weight())
        stretch       = prop.get_stretch()
        size          = str(prop.get_size_in_points())

        def lookup_name(name):
            try:
                fname = fontdict[name][style][variant][weight][stretch][size]
                verbose.report('\tfindfont cached %(name)s, %(style)s, %(variant)s, %(weight)s, %(stretch)s, %(size)s'%locals(), 'debug')
                verbose.report('findfont returning %s'%fname, 'debug')
                return fname
            except KeyError:
                pass

            fname = None
            font = fontdict
            if font.has_key(name):
                font = font[name]
            else:
                verbose.report('\tfindfont failed %(name)s'%locals(), 'debug')
                return None

            if font.has_key(style):
                font = font[style]
            elif style == 'italic' and font.has_key('oblique'):
                font = font['oblique']
            elif style == 'oblique' and font.has_key('italic'):
                font = font['italic']
            else:
                verbose.report('\tfindfont failed %(name)s, %(style)s'%locals(), 'debug')
                return None

            if font.has_key(variant):
                font = font[variant]
            else:
                verbose.report('\tfindfont failed %(name)s, %(style)s, %(variant)s'%locals(), 'debug')
                return None

            if not font.has_key(weight):
                setWeights(font)
            if not font.has_key(weight):
                return None
            font = font[weight]

            if font.has_key(stretch):
                stretch_font = font[stretch]
                if stretch_font.has_key('scalable'):
                    fname = stretch_font['scalable']
                elif stretch_font.has_key(size):
                    fname = stretch_font[size]

            if fname is None:
                for val in font.values():
                    if val.has_key('scalable'):
                        fname = val['scalable']
                        break

            if fname is None:
                for val in font.values():
                    if val.has_key(size):
                        fname = val[size]
                        break

            if fname is None:
                verbose.report('\tfindfont failed %(name)s, %(style)s, %(variant)s %(weight)s, %(stretch)s'%locals(), 'debug')
            else:
                fontkey = FontKey(",".join(prop.get_family()), style, variant, weight, stretch, size)
                add_filename(fontdict, fontkey, fname)
                verbose.report('\tfindfont found %(name)s, %(style)s, %(variant)s %(weight)s, %(stretch)s, %(size)s'%locals(), 'debug')
                verbose.report('findfont returning %s'%fname, 'debug')
            return fname

        font_family_aliases = Set(['serif', 'sans-serif', 'cursive',
                                   'fantasy', 'monospace', 'sans'])

        for name in prop.get_family():
            if name in font_family_aliases:
                if name == 'sans':
                    name = 'sans-serif'
                for name2 in rcParams['font.' + name]:
                    fname = lookup_name(name2)
                    if fname:
                        break
            else:
                fname = lookup_name(name)
            if fname:
                break

        if not fname:
            fontkey = FontKey(",".join(prop.get_family()), style, variant, weight, stretch, size)
            add_filename(fontdict, fontkey, self.defaultFont)
            verbose.report('Could not match %s, %s, %s.  Returning %s' % (name, style, weight, self.defaultFont))
            return self.defaultFont
        return fname


_is_opentype_cff_font_cache = {}
def is_opentype_cff_font(filename):
    """
    Returns True if the given font is a Postscript Compact Font Format
    Font embedded in an OpenType wrapper.
    """
    if os.path.splitext(filename)[1].lower() == '.otf':
        result = _is_opentype_cff_font_cache.get(filename)
        if result is None:
            fd = open(filename, 'rb')
            tag = fd.read(4)
            fd.close()
            result = (tag == 'OTTO')
            _is_opentype_cff_font_cache[filename] = result
        return result
    return False


if USE_FONTCONFIG and sys.platform != 'win32':
    import re

    def fc_match(pattern, fontext):
        import commands
        fontexts = get_fontext_synonyms(fontext)
        ext = "." + fontext
        status, output = commands.getstatusoutput('fc-match -sv "%s"' % pattern)
        if status == 0:
            for match in _fc_match_regex.finditer(output):
                file = match.group(1)
                if os.path.splitext(file)[1][1:] in fontexts:
                    return file
        return None

    _fc_match_regex = re.compile(r'\sfile:\s+"([^"]*)"')
    _fc_match_cache = {}

    def findfont(prop, fontext='ttf'):
        if not is_string_like(prop):
            prop = prop.get_fontconfig_pattern()
        cached = _fc_match_cache.get(prop)
        if cached is not None:
            return cached

        result = fc_match(prop, fontext)
        if result is None:
            result = fc_match(':', fontext)

        _fc_match_cache[prop] = result
        return result

else:
    _fmcache = os.path.join(get_configdir(), 'fontManager.cache')

    fontManager = None

    def _rebuild():
        global fontManager
        fontManager = FontManager()
        pickle_dump(fontManager, _fmcache)
        verbose.report("generated new fontManager")

    try:
        fontManager = pickle_load(_fmcache)
        fontManager.default_size = None
        verbose.report("Using fontManager instance from %s" % _fmcache)
    except:
        _rebuild()

    def findfont(prop, **kw):
        global fontManager
        font = fontManager.findfont(prop, **kw)
        if not os.path.exists(font):
            verbose.report("%s returned by pickled fontManager does not exist" % font)
            _rebuild()
            font =  fontManager.findfont(prop, **kw)
        return font

