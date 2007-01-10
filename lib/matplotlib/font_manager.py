"""
A module for finding, managing, and using fonts across-platforms.

This module provides a single FontManager that can be shared across
backends and platforms.  The findfont() method returns the best
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
import matplotlib
from matplotlib import afm
from matplotlib import ft2font
from matplotlib import rcParams, get_data_path, get_home, get_configdir


verbose = matplotlib.verbose

font_scalings = {'xx-small': 0.579, 'x-small': 0.694, 'small': 0.833,
                 'medium': 1.0, 'large': 1.200, 'x-large': 1.440,
                 'xx-large': 1.728}

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

home = os.environ.get('HOME')
if home is not None:
    # user fonts on OSX
    path = os.path.join(home, 'Library', 'Fonts')
    OSXFontDirectories.append(path)

def win32FontDirectory():
    """Return the user-specified font directory for Win32."""

    try:
        import _winreg
    except ImportError:
        return os.path.join(os.environ['WINDIR'], 'Fonts')
    else:
        user = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, MSFolders)
        try:
            return _winreg.QueryValueEx(user, 'Fonts')[0]
        finally:
            _winreg.CloseKey(user)
    return None

def win32InstalledFonts(directory=None, fontext='ttf'):

    """Search for fonts in the specified font directory, or use the
system directories if none given.  A list of TrueType fonts are
returned by default with AFM fonts as an option.
"""

    import _winreg
    if directory is None:
        directory = win32FontDirectory()

    key, items = None, {}
    for fontdir in MSFontDirectories:
        try:
            local = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, fontdir)
        except OSError:
            continue

        if not local:
            return glob.glob(os.path.join(directory, '*.'+fontext))
        try:
            for j in range(_winreg.QueryInfoKey(local)[1]):
                try:
                    key, direc, any = _winreg.EnumValue( local, j)
                    if not os.path.dirname(direc):
                        direc = os.path.join(directory, direc)
                    direc = os.path.abspath(direc).lower()
                    if direc[-4:] == '.'+fontext:
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

def OSXInstalledFonts(directory=None, fontext=None):
    """Get list of font files on OS X - ignores font suffix by default"""
    if directory is None:
        directory = OSXFontDirectory()

    files = []
    for path in directory:
        if fontext is None:
            files.extend(glob.glob(os.path.join(path,'*')))
        else:
            files.extend(glob.glob(os.path.join(path, '*.'+fontext)))
            files.extend(glob.glob(os.path.join(path, '*.'+fontext.upper())))
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

def findSystemFonts(fontpaths=None, fontext='ttf'):

    """Search for fonts in the specified font paths, or use the system
paths if none given.  A list of TrueType fonts are returned by default
with AFM fonts as an option.
"""

    fontfiles = {}

    if fontpaths is None:

        if sys.platform == 'win32':
            fontdir = win32FontDirectory()

            fontpaths = [fontdir]
            # now get all installed fonts directly...
            for f in win32InstalledFonts(fontdir):
                base, ext = os.path.splitext(f)
                if len(ext)>1 and ext[1:].lower()==fontext:
                    fontfiles[f] = 1
        else:
            fontpaths = x11FontDirectory()
            # check for OS X & load its fonts if present
            if sys.platform == 'darwin':
                for f in OSXInstalledFonts():
                    fontfiles[f] = 1

    elif isinstance(fontpaths, (str, unicode)):
        fontpaths = [fontpaths]

    for path in fontpaths:
        files = glob.glob(os.path.join(path, '*.'+fontext))
        files.extend(glob.glob(os.path.join(path, '*.'+fontext.upper())))
        for fname in files:
            fontfiles[os.path.abspath(fname)] = 1

    return [fname for fname in fontfiles.keys() if os.path.exists(fname)]

def weight_as_number(weight):
    """Return the weight property as a numeric value.  String values
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
    """A class for storing Font properties.  It is used when populating
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
    """A function for populating the FontKey by extracting information
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
    """A function for populating the FontKey by extracting information
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
    """A function to add a font file name to the font dictionary using
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
    """A function to create a dictionary of font file paths.  The
default is to create a dictionary for TrueType fonts.  An AFM font
dictionary can optionally be created.
"""

    fontdict = {}
    #  Add fonts from list of known font files.
    seen = {}
    for fpath in fontfiles:
        verbose.report('createFontDict: %s' % (fpath), 'debug')
        fname = fpath.split('/')[-1]
        if seen.has_key(fname):  continue
        else: seen[fname] = 1
        if fontext == 'afm':
            try:
                font = afm.AFM(file(fpath))
            except RuntimeError:
                verbose.report("Could not open font file %s"%fpath)
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

        #  !!!!  Default font algorithm needs improvement
        if   prop.name.lower() in ['bitstream vera serif', 'times']:
            prop.name = 'serif'
            add_filename(fontdict, prop, fpath)
        elif prop.name.lower() in ['bitstream vera sans', 'helvetica']:
            prop.name = 'sans-serif'
            add_filename(fontdict, prop, fpath)
        elif prop.name.lower() in ['zapf chancery', 'itc zapf chancery']:
            prop.name = 'cursive'
            add_filename(fontdict, prop, fpath)
        elif prop.name.lower() in ['western', 'itc avant garde gothic']:
            prop.name = 'fantasy'
            add_filename(fontdict, prop, fpath)
        elif prop.name.lower() in ['bitstream vera sans mono', 'courier']:
            prop.name = 'monospace'
            add_filename(fontdict, prop, fpath)

    return fontdict

def setWeights(font):
    """A function to populate missing values in a font weight
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


class FontProperties:
    """A class for storing and manipulating font properties.

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

The preferred usage of font sizes is to use the absolute values, e.g.
large, instead of absolute font sizes, e.g. 12.  This approach allows
all text sizes to be made larger or smaller based on the font manager's
default font size, i.e. by using the set_default_size() method of the
font manager.


Examples:

  #  Load default font properties
  >>> p = FontProperties()
  >>> p.get_family()
  ['Bitstream Vera Sans', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucida', 'Arial', 'Helvetica', 'sans-serif']

  #  Change font family to 'fantasy'
  >>> p.set_family('fantasy')
  >>> p.get_family()
  ['Comic Sans MS', 'Chicago', 'Charcoal', 'Impact', 'Western', 'fantasy']

  #  Make these fonts highest priority in font family
  >>> p.set_name(['foo', 'fantasy', 'bar', 'baz'])
  Font name 'fantasy' is a font family. It is being deleted from the list.
  >>> p.get_family()
  ['foo', 'bar', 'baz', 'Comic Sans MS', 'Chicago', 'Charcoal', 'Impact', 'Western', 'fantasy']

"""

    def __init__(self,
                 family = None,
                 style  = None,
                 variant= None,
                 weight = None,
                 stretch= None,
                 size   = None,
                 fname = None, # if this is set, it's a hardcoded filename to use
                 ):


        if family is None: family = rcParams['font.'+rcParams['font.family']]
        if style is None: style  = rcParams['font.style']
        if variant is None: variant= rcParams['font.variant']
        if weight is None: weight = rcParams['font.weight']
        if stretch is None: stretch= rcParams['font.stretch']
        if size is None: size   = rcParams['font.size']

        if isinstance(family, str):
            family = [family]
        self.__family  = family
        self.__style   = style
        self.__variant = variant
        self.__weight  = weight
        self.__stretch = stretch
        self.__size    = size
        self.__parent_size = fontManager.get_default_size()
        self.fname = fname

    def __hash__(self):
        return hash( (
            tuple(self.__family), self.__style, self.__variant,
            self.__weight, self.__stretch, self.__size,
            self.__parent_size, self.fname))

    def __str__(self):
        return str((self.__family, self.__style, self.__variant,
                    self.__weight, self.__stretch, self.__size))

    def get_family(self):
        """Return a list of font names that comprise the font family.
        """
        return self.__family

    def get_name(self):
        """Return the name of the font that best matches the font
properties.
"""
        return ft2font.FT2Font(str(fontManager.findfont(self))).family_name

    def get_style(self):
        """Return the font style.  Values are: normal, italic or oblique.
        """
        return self.__style

    def get_variant(self):
        """Return the font variant.  Values are: normal or small-caps.
        """
        return self.__variant

    def get_weight(self):
        """Return the font weight.  See the FontProperties class for a
a list of possible values.
"""
        return self.__weight

    def get_stretch(self):
        """Return the font stretch or width.  Options are: normal,
narrow, condensed, or wide.
"""
        return self.__stretch

    def get_size(self):
        """Return the font size.
        """
        return self.__size

    def set_family(self, family):
        """Change the font family.  Options are: serif, sans-serif, cursive,
fantasy, or monospace."""

        try:
            self.__family = rcParams['font.'+family]
            if isinstance(self.__family, str):
                self.__family = [self.__family]
        except KeyError:
            raise KeyError, '%s - use serif, sans-serif, cursive, fantasy, or monospace.' % family

    def set_name(self, names):

        """Add one or more font names to the font family list.  If the
font name is already in the list, then the font is given a higher
priority in the font family list.  To change the font family, use the
set_family() method.
"""

        msg = "Font name '%s' is a font family. It is being deleted from the list."
        font_family = ['serif', 'sans-serif', 'cursive', 'fantasy',
                       'monospace']

        if isinstance(names, str):
            names = [names]

        #  Remove family names from list of font names.
        for name in names[:]:
            if name.lower() in font_family:
                verbose.report( msg % name)
                while name in names:
                    names.remove(name.lower())

        #  Remove font names from family list.
        for name in names:
            while name in self.__family:
                self.__family.remove(name)

        self.__family = names + self.__family

    def set_style(self, style):
        """Set the font style.  Values are: normal, italic or oblique.
        """
        self.__style = style

    def set_variant(self, variant):
        """Set the font variant.  Values are: normal or small-caps.
        """
        self.__variant = variant

    def set_weight(self, weight):
        """Set the font weight.  See the FontProperties class for a
a list of possible values.
"""
        self.__weight = weight

    def set_stretch(self, stretch):
        """Set the font stretch or width.  Options are: normal, narrow,
condensed, or wide.
"""
        self.__stretch = stretch

    def set_size(self, size):
        """Set the font size.
        """
        self.__size = size

    def get_size_in_points(self, parent_size=None):
        """Return the size property as a numeric value.  String values
are converted to their corresponding numeric value.
"""
        if self.__size in font_scalings.keys():
            size = fontManager.get_default_size()*font_scalings[self.__size]
        elif self.__size == 'larger':
            size = self.__parent_size*1.2
        elif self.__size == 'smaller':
            size = self.__parent_size/1.2
        else:
            size = self.__size
        return float(size)

    def copy(self):
        """Return a deep copy of self"""
        return FontProperties(self.__family,
                              self.__style,
                              self.__variant,
                              self.__weight,
                              self.__stretch,
                              self.__size)

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

class FontManager:

    """On import, the FontManager creates a dictionary of TrueType
fonts based on the font properties: name, style, variant, weight,
stretch, and size.  The findfont() method searches this dictionary
for a font file name that exactly matches the font properties of the
specified text.  If none is found, a default font is returned.  By
updating the dictionary with the properties of the found font, the
font dictionary can act like a font cache.
"""

    def __init__(self, size=None, weight='normal'):
        if not size : size = rcParams['font.size']
        self.__default_size = size
        self.__default_weight = weight

        paths = [rcParams['datapath']]

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

        cache_message = \
"""Saving TTF font cache for non-PS backends to %s.
Delete this file to have matplotlib rebuild the cache."""



        oldcache = os.path.join(get_home(), 'ttffont.cache')
        ttfcache = os.path.join(get_configdir(), 'ttffont.cache')
        if os.path.exists(oldcache):
            print >> sys.stderr, 'Moving old ttfcache location "%s" to new location "%s"'%(oldcache, ttfcache)
            shutil.move(oldcache, ttfcache)



        try:
            import cPickle as pickle
        except ImportError:
            import pickle


        def rebuild():
            self.ttfdict = createFontDict(self.ttffiles)
            pickle.dump(self.ttfdict, file(ttfcache, 'w'))
            verbose.report(cache_message % ttfcache)

        try:
            self.ttfdict = pickle.load(file(ttfcache))
        except:
            rebuild()
        else:
            # verify all the cached fnames still exist; if not rebuild
            for fname in ttfdict_to_fnames(self.ttfdict):
                if not os.path.exists(fname):
                    rebuild()
                    break
            verbose.report('loaded ttfcache file %s'%ttfcache)



        #self.ttfdict = createFontDict(self.ttffiles)

        #  Load AFM fonts for PostScript
        #  Only load file names at this stage, the font dictionary will be
        #  created when needed.

        self.afmfiles = findSystemFonts(paths, fontext='afm') + \
                        findSystemFonts(fontext='afm')
        self.afmdict = {}

    def get_default_weight(self):
        "Return the default font weight."
        return self.__default_weight

    def get_default_size(self):
        "Return the default font size."
        return self.__default_size

    def set_default_weight(self, weight):
        "Set the default font weight.  The initial value is 'normal'."
        self.__default_weight = weight

    def set_default_size(self, size):
        "Set the default font size in points.  The initial value is set by font.size in rc."
        self.__default_size = size

    def update_fonts(self, filenames):
        """Update the font dictionary with new font files.
Currently not implemented."""
        #  !!!!  Needs implementing
        raise NotImplementedError


    def findfont(self, prop, fontext='ttf'):

        """Search the font dictionary for a font that exactly or closely
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
        cache_message = \
"""Saving AFM font cache for PS backend to %s.
Delete this file to have matplotlib rebuild the cache."""

        debug = False
        if prop.fname is not None:
            fname = prop.fname
            verbose.report('findfont returning %s'%fname, 'debug')
            return fname

        if fontext == 'afm':
            if len(self.afmdict) == 0:
                afmcache = os.path.join(get_configdir(), '.afmfont.cache')
                try:
                    import cPickle as pickle
                except ImportError:
                    import pickle
                try:
                    self.afmdict = pickle.load(file(afmcache))
                except:
                    self.afmdict = createFontDict(self.afmfiles, fontext='afm')
                    pickle.dump(self.afmdict, file(afmcache, 'w'))
                    verbose.report(cache_message % afmcache)
            fontdict = self.afmdict
        else:
            fontdict = self.ttfdict

        name    = prop.get_family()[0]
        style   = prop.get_style()
        variant = prop.get_variant()
        weight  = weight_as_number(prop.get_weight())
        stretch = prop.get_stretch()
        size    = str(prop.get_size_in_points())


        try:
            fname = fontdict[name][style][variant][weight][stretch][size]
            verbose.report('\tfindfont cached %(name)s, %(style)s, %(variant)s, %(weight)s, %(stretch)s, %(size)s'%locals(), 'debug')
            verbose.report('findfont returning %s'%fname, 'debug')
            return fname
        except KeyError:
            pass

        for name in prop.get_family():
            font = fontdict
            if font.has_key(name):
                font = font[name]
            else:
                verbose.report('\tfindfont failed %(name)s'%locals(), 'debug')
                continue

            if font.has_key(style):
                font = font[style]
            elif style == 'italics' and font.has_key('oblique'):
                font = font['oblique']
            else:
                verbose.report('\tfindfont failed %(name)s, %(style)s'%locals(), 'debug')
                continue

            if font.has_key(variant):
                font = font[variant]
            else:
                verbose.report('\tfindfont failed %(name)s, %(style)s, %(variant)s'%locals(), 'debug')
                continue

            if not font.has_key(weight):
                setWeights(font)
            font = font[weight]

            # !!!!  need improvement
            if font.has_key(stretch):
                font = font[stretch]
            else:
                verbose.report('\tfindfont failed %(name)s, %(style)s, %(variant)s %(weight)s, %(stretch)s'%locals(), 'debug')
                continue

            if font.has_key('scalable'):
                fname = font['scalable']
            elif font.has_key(size):
                fname = font[size]
            else:
                verbose.report('\tfindfont failed %(name)s, %(style)s, %(variant)s %(weight)s, %(stretch)s, %(size)s'%locals(), 'debug')
                continue

            fontkey = FontKey(name, style, variant, weight, stretch, size)
            add_filename(fontdict, fontkey, fname)
            verbose.report('\tfindfont found %(name)s, %(style)s, %(variant)s %(weight)s, %(stretch)s, %(size)s'%locals(), 'debug')
            verbose.report('findfont returning %s'%fname, 'debug')

            return fname

        fontkey = FontKey(name, style, variant, weight, stretch, size)
        add_filename(fontdict, fontkey, self.defaultFont)
        verbose.report('Could not match %s, %s, %s.  Returning %s' % (name, style, variant, self.defaultFont))

        return self.defaultFont

fontManager = FontManager()
