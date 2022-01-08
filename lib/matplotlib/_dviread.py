"""
A module for reading dvi files output by TeX. Several limitations make
this not (currently) useful as a general-purpose dvi preprocessor, but
it is currently used by the pdf backend for processing usetex text.

Interface::

  with Dvi(filename, 72) as dvi:
      # iterate over pages:
      for page in dvi:
          w, h, d = page.width, page.height, page.descent
          for x, y, font, glyph, width in page.text:
              fontname = font.texname
              pointsize = font.size
              ...
          for x, y, height, width in page.boxes:
              ...
"""

from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

from matplotlib import _api, cbook

_log = logging.getLogger(__name__)

# Many dvi related files are looked for by external processes, require
# additional parsing, and are used many times per rendering, which is why they
# are cached using lru_cache().

# The marks on a page consist of text and boxes. A page also has dimensions.
Page = namedtuple('Page', 'text boxes height width descent')
Text = namedtuple('Text', 'x y font glyph width')
Box = namedtuple('Box', 'x y height width')


# Opcode argument parsing
#
# Each of the following functions takes a Dvi object and delta,
# which is the difference between the opcode and the minimum opcode
# with the same meaning. Dvi opcodes often encode the number of
# argument bytes in this delta.

def _arg_raw(dvi, delta):
    """Return *delta* without reading anything more from the dvi file."""
    return delta


def _arg(nbytes, signed, dvi, _):
    """
    Read *nbytes* bytes, returning the bytes interpreted as a signed integer
    if *signed* is true, unsigned otherwise.
    """
    return dvi._arg(nbytes, signed)


def _arg_slen(dvi, delta):
    """
    Read *delta* bytes, returning None if *delta* is zero, and the bytes
    interpreted as a signed integer otherwise.
    """
    if delta == 0:
        return None
    return dvi._arg(delta, True)


def _arg_slen1(dvi, delta):
    """
    Read *delta*+1 bytes, returning the bytes interpreted as signed.
    """
    return dvi._arg(delta + 1, True)


def _arg_ulen1(dvi, delta):
    """
    Read *delta*+1 bytes, returning the bytes interpreted as unsigned.
    """
    return dvi._arg(delta + 1, False)


def _arg_olen1(dvi, delta):
    """
    Read *delta*+1 bytes, returning the bytes interpreted as
    unsigned integer for 0<=*delta*<3 and signed if *delta*==3.
    """
    return dvi._arg(delta + 1, delta == 3)


_arg_mapping = dict(raw=_arg_raw,
                    u1=partial(_arg, 1, False),
                    u4=partial(_arg, 4, False),
                    s4=partial(_arg, 4, True),
                    slen=_arg_slen,
                    olen1=_arg_olen1,
                    slen1=_arg_slen1,
                    ulen1=_arg_ulen1)


def _dispatch(table, min, max=None, state=None, args=('raw',)):
    """
    Decorator for dispatch by opcode. Sets the values in *table*
    from *min* to *max* to this method, adds a check that the Dvi state
    matches *state* if not None, reads arguments from the file according
    to *args*.

    Parameters
    ----------
    table : dict[int, callable]
        The dispatch table to be filled in.

    min, max : int
        Range of opcodes that calls the registered function; *max* defaults to
        *min*.

    state : _dvistate, optional
        State of the Dvi object in which these opcodes are allowed.

    args : list[str], default: ['raw']
        Sequence of argument specifications:

        - 'raw': opcode minus minimum
        - 'u1': read one unsigned byte
        - 'u4': read four bytes, treat as an unsigned number
        - 's4': read four bytes, treat as a signed number
        - 'slen': read (opcode - minimum) bytes, treat as signed
        - 'slen1': read (opcode - minimum + 1) bytes, treat as signed
        - 'ulen1': read (opcode - minimum + 1) bytes, treat as unsigned
        - 'olen1': read (opcode - minimum + 1) bytes, treat as unsigned
          if under four bytes, signed if four bytes
    """
    def decorate(method):
        get_args = [_arg_mapping[x] for x in args]

        @wraps(method)
        def wrapper(self, byte):
            if state is not None and self.state != state:
                raise ValueError("state precondition failed")
            return method(self, *[f(self, byte-min) for f in get_args])
        if max is None:
            table[min] = wrapper
        else:
            for i in range(min, max+1):
                assert table[i] is None
                table[i] = wrapper
        return wrapper
    return decorate


class DviFont:
    """
    Encapsulation of a font that a DVI file can refer to.

    This class holds a font's texname and size, supports comparison,
    and knows the widths of glyphs in the same units as the AFM file.
    There are also internal attributes (for use by dviread.py) that
    are *not* used for comparison.

    The size is in Adobe points (converted from TeX points).

    Parameters
    ----------
    scale : float
        Factor by which the font is scaled from its natural size.
    tfm : Tfm
        TeX font metrics for this font
    texname : bytes
       Name of the font as used internally by TeX and friends, as an ASCII
       bytestring.  This is usually very different from any external font
       names; `PsfontsMap` can be used to find the external name of the font.
    vf : Vf
       A TeX "virtual font" file, or None if this font is not virtual.

    Attributes
    ----------
    texname : bytes
    size : float
       Size of the font in Adobe points, converted from the slightly
       smaller TeX points.
    widths : list
       Widths of glyphs in glyph-space units, typically 1/1000ths of
       the point size.

    """
    __slots__ = ('texname', 'size', 'widths', '_scale', '_vf', '_tfm')

    def __init__(self, scale, tfm, texname, vf):
        _api.check_isinstance(bytes, texname=texname)
        self._scale = scale
        self._tfm = tfm
        self.texname = texname
        self._vf = vf
        self.size = scale * (72.0 / (72.27 * 2**16))
        try:
            nchars = max(tfm.width) + 1
        except ValueError:
            nchars = 0
        self.widths = [(1000*tfm.width.get(char, 0)) >> 20
                       for char in range(nchars)]

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.texname == other.texname and self.size == other.size)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "<{}: {}>".format(type(self).__name__, self.texname)

    def _width_of(self, char):
        """Width of char in dvi units."""
        width = self._tfm.width.get(char, None)
        if width is not None:
            return _mul2012(width, self._scale)
        _log.debug('No width for char %d in font %s.', char, self.texname)
        return 0

    def _height_depth_of(self, char):
        """Height and depth of char in dvi units."""
        result = []
        for metric, name in ((self._tfm.height, "height"),
                             (self._tfm.depth, "depth")):
            value = metric.get(char, None)
            if value is None:
                _log.debug('No %s for char %d in font %s',
                           name, char, self.texname)
                result.append(0)
            else:
                result.append(_mul2012(value, self._scale))
        # cmsyXX (symbols font) glyph 0 ("minus") has a nonzero descent
        # so that TeX aligns equations properly
        # (https://tex.stackexchange.com/q/526103/)
        # but we actually care about the rasterization depth to align
        # the dvipng-generated images.
        if re.match(br'^cmsy\d+$', self.texname) and char == 0:
            result[-1] = 0
        return result


def _mul2012(num1, num2):
    """Multiply two numbers in 20.12 fixed point format."""
    # Separated into a function because >> has surprising precedence
    return (num1*num2) >> 20


class Tfm:
    """
    A TeX Font Metric file.

    This implementation covers only the bare minimum needed by the Dvi class.

    Parameters
    ----------
    filename : str or path-like

    Attributes
    ----------
    checksum : int
       Used for verifying against the dvi file.
    design_size : int
       Design size of the font (unknown units)
    width, height, depth : dict
       Dimensions of each character, need to be scaled by the factor
       specified in the dvi file. These are dicts because indexing may
       not start from 0.
    """
    __slots__ = ('checksum', 'design_size', 'width', 'height', 'depth')

    def __init__(self, filename):
        _log.debug('opening tfm file %s', filename)
        with open(filename, 'rb') as file:
            header1 = file.read(24)
            lh, bc, ec, nw, nh, nd = struct.unpack('!6H', header1[2:14])
            _log.debug('lh=%d, bc=%d, ec=%d, nw=%d, nh=%d, nd=%d',
                       lh, bc, ec, nw, nh, nd)
            header2 = file.read(4*lh)
            self.checksum, self.design_size = struct.unpack('!2I', header2[:8])
            # there is also encoding information etc.
            char_info = file.read(4*(ec-bc+1))
            widths = struct.unpack(f'!{nw}i', file.read(4*nw))
            heights = struct.unpack(f'!{nh}i', file.read(4*nh))
            depths = struct.unpack(f'!{nd}i', file.read(4*nd))
        self.width, self.height, self.depth = {}, {}, {}
        for idx, char in enumerate(range(bc, ec+1)):
            byte0 = char_info[4*idx]
            byte1 = char_info[4*idx+1]
            self.width[char] = widths[byte0]
            self.height[char] = heights[byte1 >> 4]
            self.depth[char] = depths[byte1 & 0xf]


PsFont = namedtuple('PsFont', 'texname psname effects encoding filename')


class PsfontsMap:
    """
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
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
    :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
    The file :file:`psfonts.map` is used by :program:`dvips`,
    :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
    by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
    the 35 PostScript fonts (i.e., have no filename for them, as in
    the Times-Bold example above), while the pdf-related files perhaps
    only avoid the "Base 14" pdf fonts. But the user may have
    configured these files differently.

    Examples
    --------
    >>> map = PsfontsMap(find_tex_file('pdftex.map'))
    >>> entry = map[b'ptmbo8r']
    >>> entry.texname
    b'ptmbo8r'
    >>> entry.psname
    b'Times-Bold'
    >>> entry.encoding
    '/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
    >>> entry.effects
    {'slant': 0.16700000000000001}
    >>> entry.filename
    """
    __slots__ = ('_filename', '_unparsed', '_parsed')

    # Create a filename -> PsfontsMap cache, so that calling
    # `PsfontsMap(filename)` with the same filename a second time immediately
    # returns the same object.
    @lru_cache()
    def __new__(cls, filename):
        self = object.__new__(cls)
        self._filename = os.fsdecode(filename)
        # Some TeX distributions have enormous pdftex.map files which would
        # take hundreds of milliseconds to parse, but it is easy enough to just
        # store the unparsed lines (keyed by the first word, which is the
        # texname) and parse them on-demand.
        with open(filename, 'rb') as file:
            self._unparsed = {}
            for line in file:
                tfmname = line.split(b' ', 1)[0]
                self._unparsed.setdefault(tfmname, []).append(line)
        self._parsed = {}
        return self

    def __getitem__(self, texname):
        assert isinstance(texname, bytes)
        if texname in self._unparsed:
            for line in self._unparsed.pop(texname):
                if self._parse_and_cache_line(line):
                    break
        try:
            return self._parsed[texname]
        except KeyError:
            raise LookupError(
                f"An associated PostScript font (required by Matplotlib) "
                f"could not be found for TeX font {texname.decode('ascii')!r} "
                f"in {self._filename!r}; this problem can often be solved by "
                f"installing a suitable PostScript font package in your TeX "
                f"package manager") from None

    def _parse_and_cache_line(self, line):
        """
        Parse a line in the font mapping file.

        The format is (partially) documented at
        http://mirrors.ctan.org/systems/doc/pdftex/manual/pdftex-a.pdf
        https://tug.org/texinfohtml/dvips.html#psfonts_002emap
        Each line can have the following fields:

        - tfmname (first, only required field),
        - psname (defaults to tfmname, must come immediately after tfmname if
          present),
        - fontflags (integer, must come immediately after psname if present,
          ignored by us),
        - special (SlantFont and ExtendFont, only field that is double-quoted),
        - fontfile, encodingfile (optional, prefixed by <, <<, or <[; << always
          precedes a font, <[ always precedes an encoding, < can precede either
          but then an encoding file must have extension .enc; < and << also
          request different font subsetting behaviors but we ignore that; < can
          be separated from the filename by whitespace).

        special, fontfile, and encodingfile can appear in any order.
        """
        # If the map file specifies multiple encodings for a font, we
        # follow pdfTeX in choosing the last one specified. Such
        # entries are probably mistakes but they have occurred.
        # https://tex.stackexchange.com/q/10826/

        if not line or line.startswith((b" ", b"%", b"*", b";", b"#")):
            return
        tfmname = basename = special = encodingfile = fontfile = None
        is_subsetted = is_t1 = is_truetype = False
        matches = re.finditer(br'"([^"]*)(?:"|$)|(\S+)', line)
        for match in matches:
            quoted, unquoted = match.groups()
            if unquoted:
                if unquoted.startswith(b"<<"):  # font
                    fontfile = unquoted[2:]
                elif unquoted.startswith(b"<["):  # encoding
                    encodingfile = unquoted[2:]
                elif unquoted.startswith(b"<"):  # font or encoding
                    word = (
                        # <foo => foo
                        unquoted[1:]
                        # < by itself => read the next word
                        or next(filter(None, next(matches).groups())))
                    if word.endswith(b".enc"):
                        encodingfile = word
                    else:
                        fontfile = word
                        is_subsetted = True
                elif tfmname is None:
                    tfmname = unquoted
                elif basename is None:
                    basename = unquoted
            elif quoted:
                special = quoted
        effects = {}
        if special:
            words = reversed(special.split())
            for word in words:
                if word == b"SlantFont":
                    effects["slant"] = float(next(words))
                elif word == b"ExtendFont":
                    effects["extend"] = float(next(words))

        # Verify some properties of the line that would cause it to be ignored
        # otherwise.
        if fontfile is not None:
            if fontfile.endswith((b".ttf", b".ttc")):
                is_truetype = True
            elif not fontfile.endswith(b".otf"):
                is_t1 = True
        elif basename is not None:
            is_t1 = True
        if is_truetype and is_subsetted and encodingfile is None:
            return
        if not is_t1 and ("slant" in effects or "extend" in effects):
            return
        if abs(effects.get("slant", 0)) > 1:
            return
        if abs(effects.get("extend", 0)) > 2:
            return

        if basename is None:
            basename = tfmname
        if encodingfile is not None:
            encodingfile = _find_tex_file(encodingfile)
        if fontfile is not None:
            fontfile = _find_tex_file(fontfile)
        self._parsed[tfmname] = PsFont(
            texname=tfmname, psname=basename, effects=effects,
            encoding=encodingfile, filename=fontfile)
        return True


def _parse_enc(path):
    r"""
    Parse a \*.enc file referenced from a psfonts.map style file.

    The format supported by this function is a tiny subset of PostScript.

    Parameters
    ----------
    path : os.PathLike

    Returns
    -------
    list
        The nth entry of the list is the PostScript glyph name of the nth
        glyph.
    """
    no_comments = re.sub("%.*", "", Path(path).read_text(encoding="ascii"))
    array = re.search(r"(?s)\[(.*)\]", no_comments).group(1)
    lines = [line for line in array.split() if line]
    if all(line.startswith("/") for line in lines):
        return [line[1:] for line in lines]
    else:
        raise ValueError(
            "Failed to parse {} as Postscript encoding".format(path))


class _LuatexKpsewhich:
    @lru_cache()  # A singleton.
    def __new__(cls):
        self = object.__new__(cls)
        self._proc = self._new_proc()
        return self

    def _new_proc(self):
        return subprocess.Popen(
            ["luatex", "--luaonly",
             str(cbook._get_data_path("kpsewhich.lua"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def search(self, filename):
        if self._proc.poll() is not None:  # Dead, restart it.
            self._proc = self._new_proc()
        self._proc.stdin.write(os.fsencode(filename) + b"\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline().rstrip()
        return None if out == b"nil" else os.fsdecode(out)


@lru_cache()
@_api.delete_parameter("3.5", "format")
def _find_tex_file(filename, format=None):
    """
    Find a file in the texmf tree using kpathsea_.

    The kpathsea library, provided by most existing TeX distributions, both
    on Unix-like systems and on Windows (MikTeX), is invoked via a long-lived
    luatex process if luatex is installed, or via kpsewhich otherwise.

    .. _kpathsea: https://www.tug.org/kpathsea/

    Parameters
    ----------
    filename : str or path-like
    format : str or bytes
        Used as the value of the ``--format`` option to :program:`kpsewhich`.
        Could be e.g. 'tfm' or 'vf' to limit the search to that type of files.
        Deprecated.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """

    # we expect these to always be ascii encoded, but use utf-8
    # out of caution
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='replace')
    if isinstance(format, bytes):
        format = format.decode('utf-8', errors='replace')

    try:
        lk = _LuatexKpsewhich()
    except FileNotFoundError:
        lk = None  # Fallback to directly calling kpsewhich, as below.

    if lk and format is None:
        path = lk.search(filename)

    else:
        if os.name == 'nt':
            # On Windows only, kpathsea can use utf-8 for cmd args and output.
            # The `command_line_encoding` environment variable is set to force
            # it to always use utf-8 encoding.  See Matplotlib issue #11848.
            kwargs = {'env': {**os.environ, 'command_line_encoding': 'utf-8'},
                      'encoding': 'utf-8'}
        else:  # On POSIX, run through the equivalent of os.fsdecode().
            kwargs = {'encoding': sys.getfilesystemencoding(),
                      'errors': 'surrogateescape'}

        cmd = ['kpsewhich']
        if format is not None:
            cmd += ['--format=' + format]
        cmd += [filename]
        try:
            path = (cbook._check_and_log_subprocess(cmd, _log, **kwargs)
                    .rstrip('\n'))
        except (FileNotFoundError, RuntimeError):
            path = None

    if path:
        return path
    else:
        raise FileNotFoundError(
            f"Matplotlib's TeX implementation searched for a file named "
            f"{filename!r} in your texmf tree, but could not find it")


# After the deprecation period elapses, delete this shim and rename
# _find_tex_file to find_tex_file everywhere.
@_api.delete_parameter("3.5", "format")
def find_tex_file(filename, format=None):
    try:
        return (_find_tex_file(filename, format) if format is not None else
                _find_tex_file(filename))
    except FileNotFoundError as exc:
        _api.warn_deprecated(
            "3.6", message=f"{exc.args[0]}; in the future, this will raise a "
            f"FileNotFoundError.")
        return ""


find_tex_file.__doc__ = _find_tex_file.__doc__


@lru_cache()
def _fontfile(cls, suffix, texname):
    return cls(_find_tex_file(texname + suffix))


_tfmfile = partial(_fontfile, Tfm, ".tfm")
