"""
Common functionality between the PDF and PS backends.
"""

from __future__ import annotations

from io import BytesIO
import functools
import logging
import typing

from fontTools import subset

import matplotlib as mpl
from .. import font_manager, ft2font
from .._afm import AFM
from ..backend_bases import RendererBase


if typing.TYPE_CHECKING:
    from .ft2font import CharacterCodeType, FT2Font, GlyphIndexType
    from fontTools.ttLib import TTFont


@functools.lru_cache(50)
def _cached_get_afm_from_fname(fname):
    with open(fname, "rb") as fh:
        return AFM(fh)


def get_glyphs_subset(fontfile: str, glyphs: set[GlyphIndexType]) -> TTFont:
    """
    Subset a TTF font.

    Reads the named fontfile and restricts the font to the glyphs.

    Parameters
    ----------
    fontfile : str
        Path to the font file
    glyphs : set[GlyphIndexType]
        Set of glyph indices to include in subset.

    Returns
    -------
    fontTools.ttLib.ttFont.TTFont
        An open font object representing the subset, which needs to
        be closed by the caller.
    """
    options = subset.Options(glyph_names=True, recommended_glyphs=True,
                             retain_gids=True)

    # Prevent subsetting extra tables.
    options.drop_tables += [
        'FFTM',  # FontForge Timestamp.
        'PfEd',  # FontForge personal table.
        'BDF',  # X11 BDF header.
        'meta',  # Metadata stores design/supported languages (meaningless for subsets).
        'MERG',  # Merge Table.
        'TSIV',  # Microsoft Visual TrueType extension.
        'Zapf',  # Information about the individual glyphs in the font.
        'bdat',  # The bitmap data table.
        'bloc',  # The bitmap location table.
        'cidg',  # CID to Glyph ID table (Apple Advanced Typography).
        'fdsc',  # The font descriptors table.
        'feat',  # Feature name table (Apple Advanced Typography).
        'fmtx',  # The Font Metrics Table.
        'fond',  # Data-fork font information (Apple Advanced Typography).
        'just',  # The justification table (Apple Advanced Typography).
        'kerx',  # An extended kerning table (Apple Advanced Typography).
        'ltag',  # Language Tag.
        'morx',  # Extended Glyph Metamorphosis Table.
        'trak',  # Tracking table.
        'xref',  # The cross-reference table (some Apple font tooling information).
    ]
    # if fontfile is a ttc, specify font number
    if fontfile.endswith(".ttc"):
        options.font_number = 0

    font = subset.load_font(fontfile, options)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(gids=glyphs)
    subsetter.subset(font)
    return font


def font_as_file(font):
    """
    Convert a TTFont object into a file-like object.

    Parameters
    ----------
    font : fontTools.ttLib.ttFont.TTFont
        A font object

    Returns
    -------
    BytesIO
        A file object with the font saved into it
    """
    fh = BytesIO()
    font.save(fh, reorderTables=False)
    return fh


class CharacterTracker:
    """
    Helper for font subsetting by the PDF and PS backends.

    Maintains a mapping of font paths to the set of characters and glyphs that are being
    used from that font.

    Attributes
    ----------
    subset_size : int
        The size at which characters are grouped into subsets.
    used : dict[tuple[str, int], dict[CharacterCodeType, GlyphIndexType]]
        A dictionary of font files to character maps. The key is a font filename and
        subset within that font. The value is a dictionary mapping a character code to a
        glyph index. If *subset_size* is not set, then there will only be one subset per
        font filename.
    """

    def __init__(self, subset_size: int = 0):
        """
        Parameters
        ----------
        subset_size : int, optional
            The maximum size that is supported for an embedded font. If provided, then
            characters will be grouped into these sized subsets.
        """
        self.used: dict[tuple[str, int], dict[CharacterCodeType, GlyphIndexType]] = {}
        self.subset_size = subset_size

    def track(self, font: FT2Font, s: str) -> list[tuple[int, CharacterCodeType]]:
        """
        Record that string *s* is being typeset using font *font*.

        Parameters
        ----------
        font : FT2Font
            A font that is being used for the provided string.
        s : str
            The string that should be marked as tracked by the provided font.

        Returns
        -------
        list[tuple[int, CharacterCodeType]]
            A list of subset and character code pairs corresponding to the input string.
            If a *subset_size* is specified on this instance, then the character code
            will correspond with the given subset (and not necessarily the string as a
            whole). If *subset_size* is not specified, then the subset will always be 0
            and the character codes will be returned from the string unchanged.
        """
        font_glyphs = []
        char_to_font = font._get_fontmap(s)
        for _c, _f in char_to_font.items():
            charcode = ord(_c)
            glyph_index = _f.get_char_index(charcode)
            if self.subset_size != 0:
                subset = charcode // self.subset_size
                subset_charcode = charcode % self.subset_size
            else:
                subset = 0
                subset_charcode = charcode
            self.used.setdefault((_f.fname, subset), {})[subset_charcode] = glyph_index
            font_glyphs.append((subset, subset_charcode))
        return font_glyphs

    def track_glyph(
            self, font: FT2Font, glyph: GlyphIndexType,
            charcode: CharacterCodeType | None = None) -> tuple[int, CharacterCodeType]:
        """
        Record character code *charcode* at glyph index *glyph* as using font *font*.

        Parameters
        ----------
        font : FT2Font
            A font that is being used for the provided string.
        glyph : GlyphIndexType
            The corresponding glyph index to record.
        charcode : CharacterCodeType, optional
            The character code to record. If not given, assume it's the same as the
            glyph index.

        Returns
        -------
        subset : int
            The subset in which the returned character code resides. If *subset_size*
            was not specified on this instance, then this is always 0.
        subset_charcode : CharacterCodeType
            The character code within the above subset. If *subset_size* was not
            specified on this instance, then this is just *charcode* unmodified.
        """
        if charcode is None:
            # Assume we don't care, so use a correspondingly unique value.
            charcode = typing.cast('CharacterCodeType', glyph)
        if self.subset_size != 0:
            subset = charcode // self.subset_size
            subset_charcode = charcode % self.subset_size
        else:
            subset = 0
            subset_charcode = charcode
        self.used.setdefault((font.fname, subset), {})[subset_charcode] = glyph
        return (subset, subset_charcode)

    def subset_to_unicode(self, index: int,
                          charcode: CharacterCodeType) -> CharacterCodeType:
        """
        Map a subset index and character code to a Unicode character code.

        Parameters
        ----------
        index : int
            The subset index within a font.
        charcode : CharacterCodeType
            The character code within a subset to map back.

        Returns
        -------
        CharacterCodeType
            The Unicode character code corresponding to the subsetted one.
        """
        return index * self.subset_size + charcode


class RendererPDFPSBase(RendererBase):
    # The following attributes must be defined by the subclasses:
    # - _afm_font_dir
    # - _use_afm_rc_name

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def flipy(self):
        # docstring inherited
        return False  # y increases from bottom to top.

    def option_scale_image(self):
        # docstring inherited
        return True  # PDF and PS support arbitrary image scaling.

    def option_image_nocomposite(self):
        # docstring inherited
        # Decide whether to composite image based on rcParam value.
        return not mpl.rcParams["image.composite_image"]

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width * 72.0, self.height * 72.0

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        if ismath == "TeX":
            return super().get_text_width_height_descent(s, prop, ismath)
        elif ismath:
            parse = self._text2path.mathtext_parser.parse(s, 72, prop)
            return parse.width, parse.height, parse.depth
        elif mpl.rcParams[self._use_afm_rc_name]:
            font = self._get_font_afm(prop)
            l, b, w, h, d = font.get_str_bbox_and_descent(s)
            scale = prop.get_size_in_points() / 1000
            w *= scale
            h *= scale
            d *= scale
            return w, h, d
        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0.0, flags=ft2font.LoadFlags.NO_HINTING)
            w, h = font.get_width_height()
            d = font.get_descent()
            scale = 1 / 64
            w *= scale
            h *= scale
            d *= scale
            return w, h, d

    def _get_font_afm(self, prop):
        fname = font_manager.findfont(
            prop, fontext="afm", directory=self._afm_font_dir)
        return _cached_get_afm_from_fname(fname)

    def _get_font_ttf(self, prop):
        fnames = font_manager.fontManager._find_fonts_by_props(prop)
        try:
            font = font_manager.get_font(fnames)
            font.clear()
            font.set_size(prop.get_size_in_points(), 72)
            return font
        except RuntimeError:
            logging.getLogger(__name__).warning(
                "The PostScript/PDF backend does not currently "
                "support the selected font (%s).", fnames)
            raise
