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


class GlyphMap:
    """
    A two-way glyph mapping.

    The forward glyph map is from (character code, glyph index)-pairs to (subset index,
    subset character code)-pairs.

    The inverse glyph map is from to (subset index, subset character code)-pairs to
    (character code, glyph index)-pairs.
    """

    def __init__(self) -> None:
        self._forward: dict[tuple[CharacterCodeType, GlyphIndexType],
                            tuple[int, CharacterCodeType]] = {}
        self._inverse: dict[tuple[int, CharacterCodeType],
                            tuple[CharacterCodeType, GlyphIndexType]] = {}

    def get(self, charcode: CharacterCodeType,
            glyph_index: GlyphIndexType) -> tuple[int, CharacterCodeType] | None:
        """
        Get the forward mapping from a (character code, glyph index)-pair.

        This may return *None* if the pair is not currently mapped.
        """
        return self._forward.get((charcode, glyph_index))

    def iget(self, subset: int,
             subset_charcode: CharacterCodeType) -> tuple[CharacterCodeType,
                                                          GlyphIndexType]:
        """Get the inverse mapping from a (subset, subset charcode)-pair."""
        return self._inverse[(subset, subset_charcode)]

    def add(self, charcode: CharacterCodeType, glyph_index: GlyphIndexType, subset: int,
            subset_charcode: CharacterCodeType) -> None:
        """
        Add a mapping to this instance.

        Parameters
        ----------
        charcode : CharacterCodeType
            The character code to record.
        glyph : GlyphIndexType
            The corresponding glyph index to record.
        subset : int
            The subset in which the subset character code resides.
        subset_charcode : CharacterCodeType
            The subset character code within the above subset.
        """
        self._forward[(charcode, glyph_index)] = (subset, subset_charcode)
        self._inverse[(subset, subset_charcode)] = (charcode, glyph_index)


class CharacterTracker:
    """
    Helper for font subsetting by the PDF and PS backends.

    Maintains a mapping of font paths to the set of characters and glyphs that are being
    used from that font.

    Attributes
    ----------
    subset_size : int
        The size at which characters are grouped into subsets.
    used : dict
        A dictionary of font files to character maps.

        The key is a font filename.

        The value is a list of dictionaries, each mapping at most *subset_size*
        character codes to glyph indices. Note this mapping is the inverse of FreeType,
        which maps glyph indices to character codes.

        If *subset_size* is not set, then there will only be one subset per font
        filename.
    glyph_maps : dict
        A dictionary of font files to glyph maps. You probably will want to use the
        `.subset_to_unicode` method instead of this attribute.
    """

    def __init__(self, subset_size: int = 0):
        """
        Parameters
        ----------
        subset_size : int, optional
            The maximum size that is supported for an embedded font. If provided, then
            characters will be grouped into these sized subsets.
        """
        self.used: dict[str, list[dict[CharacterCodeType, GlyphIndexType]]] = {}
        self.glyph_maps: dict[str, GlyphMap] = {}
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
        return [
            self.track_glyph(f, ord(c), f.get_char_index(ord(c)))
            for c, f in font._get_fontmap(s).items()
        ]

    def track_glyph(
            self, font: FT2Font, charcode: CharacterCodeType,
            glyph: GlyphIndexType) -> tuple[int, CharacterCodeType]:
        """
        Record character code *charcode* at glyph index *glyph* as using font *font*.

        Parameters
        ----------
        font : FT2Font
            A font that is being used for the provided string.
        charcode : CharacterCodeType
            The character code to record.
        glyph : GlyphIndexType
            The corresponding glyph index to record.

        Returns
        -------
        subset : int
            The subset in which the returned character code resides. If *subset_size*
            was not specified on this instance, then this is always 0.
        subset_charcode : CharacterCodeType
            The character code within the above subset. If *subset_size* was not
            specified on this instance, then this is just *charcode* unmodified.
        """
        glyph_map = self.glyph_maps.setdefault(font.fname, GlyphMap())
        if result := glyph_map.get(charcode, glyph):
            return result

        subset_maps = self.used.setdefault(font.fname, [{}])
        # Default to preserving the character code as it was.
        use_next_charmap = (
            self.subset_size != 0
            # But start filling a new subset if outside the first block; this preserves
            # ASCII (for Type 3) or the Basic Multilingual Plane (for Type 42).
            and charcode >= self.subset_size
        )
        if use_next_charmap:
            if len(subset_maps) == 1 or len(subset_maps[-1]) == self.subset_size:
                subset_maps.append({})
            subset = len(subset_maps) - 1
            subset_charcode = len(subset_maps[-1])
        else:
            subset = 0
            subset_charcode = charcode
        subset_maps[subset][subset_charcode] = glyph
        glyph_map.add(charcode, glyph, subset, subset_charcode)
        return (subset, subset_charcode)

    def subset_to_unicode(self, fontname: str, subset: int,
                          subset_charcode: CharacterCodeType) -> CharacterCodeType:
        """
        Map a subset index and character code to a Unicode character code.

        Parameters
        ----------
        fontname : str
            The name of the font, from the *used* dictionary key.
        subset : int
            The subset index within a font.
        subset_charcode : CharacterCodeType
            The character code within a subset to map back.

        Returns
        -------
        CharacterCodeType
            The Unicode character code corresponding to the subsetted one.
        """
        return self.glyph_maps[fontname].iget(subset, subset_charcode)[0]


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
