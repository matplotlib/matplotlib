"""GlyphSets returned by a TTFont."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import copy
from fontTools.misc.fixedTools import otRound
from fontTools.misc.loggingTools import deprecateFunction


class _TTGlyphSet(Mapping):

    """Generic dict-like GlyphSet class that pulls metrics from hmtx and
    glyph shape from TrueType or CFF.
    """

    def __init__(self, font, location, glyphsMapping):
        self.font = font
        self.location = location
        self.glyphsMapping = glyphsMapping
        self.hMetrics = font["hmtx"].metrics
        self.vMetrics = getattr(font.get("vmtx"), "metrics", None)
        if location:
            from fontTools.varLib.varStore import VarStoreInstancer

            self.hvarTable = getattr(font.get("HVAR"), "table", None)
            if self.hvarTable is not None:
                self.hvarInstancer = VarStoreInstancer(
                    self.hvarTable.VarStore, font["fvar"].axes, location
                )
            # TODO VVAR, VORG

    def __contains__(self, glyphName):
        return glyphName in self.glyphsMapping

    def __iter__(self):
        return iter(self.glyphsMapping.keys())

    def __len__(self):
        return len(self.glyphsMapping)

    @deprecateFunction(
        "use 'glyphName in glyphSet' instead", category=DeprecationWarning
    )
    def has_key(self, glyphName):
        return glyphName in self.glyphsMapping


class _TTGlyphSetGlyf(_TTGlyphSet):
    def __init__(self, font, location):
        self.glyfTable = font["glyf"]
        super().__init__(font, location, self.glyfTable)
        if location:
            self.gvarTable = font.get("gvar")

    def __getitem__(self, glyphName):
        return _TTGlyphGlyf(self, glyphName)


class _TTGlyphSetCFF(_TTGlyphSet):
    def __init__(self, font, location):
        tableTag = "CFF2" if "CFF2" in font else "CFF "
        self.charStrings = list(font[tableTag].cff.values())[0].CharStrings
        super().__init__(font, location, self.charStrings)
        self.blender = None
        if location:
            from fontTools.varLib.varStore import VarStoreInstancer

            varStore = getattr(self.charStrings, "varStore", None)
            if varStore is not None:
                instancer = VarStoreInstancer(
                    varStore.otVarStore, font["fvar"].axes, location
                )
                self.blender = instancer.interpolateFromDeltas

    def __getitem__(self, glyphName):
        return _TTGlyphCFF(self, glyphName)


class _TTGlyph(ABC):

    """Glyph object that supports the Pen protocol, meaning that it has
    .draw() and .drawPoints() methods that take a pen object as their only
    argument. Additionally there are 'width' and 'lsb' attributes, read from
    the 'hmtx' table.

    If the font contains a 'vmtx' table, there will also be 'height' and 'tsb'
    attributes.
    """

    def __init__(self, glyphSet, glyphName):
        self.glyphSet = glyphSet
        self.name = glyphName
        self.width, self.lsb = glyphSet.hMetrics[glyphName]
        if glyphSet.vMetrics is not None:
            self.height, self.tsb = glyphSet.vMetrics[glyphName]
        else:
            self.height, self.tsb = None, None
        if glyphSet.location and glyphSet.hvarTable is not None:
            varidx = (
                glyphSet.font.getGlyphID(glyphName)
                if glyphSet.hvarTable.AdvWidthMap is None
                else glyphSet.hvarTable.AdvWidthMap.mapping[glyphName]
            )
            self.width += glyphSet.hvarInstancer[varidx]
        # TODO: VVAR/VORG

    @abstractmethod
    def draw(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        raise NotImplementedError

    def drawPoints(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details
        how that works.
        """
        from fontTools.pens.pointPen import SegmentToPointPen

        self.draw(SegmentToPointPen(pen))


class _TTGlyphGlyf(_TTGlyph):
    def draw(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        glyph, offset = self._getGlyphAndOffset()
        glyph.draw(pen, self.glyphSet.glyfTable, offset)

    def drawPoints(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details
        how that works.
        """
        glyph, offset = self._getGlyphAndOffset()
        glyph.drawPoints(pen, self.glyphSet.glyfTable, offset)

    def _getGlyphAndOffset(self):
        if self.glyphSet.location and self.glyphSet.gvarTable is not None:
            glyph = self._getGlyphInstance()
        else:
            glyph = self.glyphSet.glyfTable[self.name]

        offset = self.lsb - glyph.xMin if hasattr(glyph, "xMin") else 0
        return glyph, offset

    def _getGlyphInstance(self):
        from fontTools.varLib.iup import iup_delta
        from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
        from fontTools.varLib.models import supportScalar

        glyphSet = self.glyphSet
        glyfTable = glyphSet.glyfTable
        variations = glyphSet.gvarTable.variations[self.name]
        hMetrics = glyphSet.hMetrics
        vMetrics = glyphSet.vMetrics
        coordinates, _ = glyfTable._getCoordinatesAndControls(
            self.name, hMetrics, vMetrics
        )
        origCoords, endPts = None, None
        for var in variations:
            scalar = supportScalar(glyphSet.location, var.axes)
            if not scalar:
                continue
            delta = var.coordinates
            if None in delta:
                if origCoords is None:
                    origCoords, control = glyfTable._getCoordinatesAndControls(
                        self.name, hMetrics, vMetrics
                    )
                    endPts = (
                        control[1] if control[0] >= 1 else list(range(len(control[1])))
                    )
                delta = iup_delta(delta, origCoords, endPts)
            coordinates += GlyphCoordinates(delta) * scalar

        glyph = copy(glyfTable[self.name])  # Shallow copy
        width, lsb, height, tsb = _setCoordinates(glyph, coordinates, glyfTable)
        self.lsb = lsb
        self.tsb = tsb
        if glyphSet.hvarTable is None:
            # no HVAR: let's set metrics from the phantom points
            self.width = width
            self.height = height
        return glyph


class _TTGlyphCFF(_TTGlyph):
    def draw(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        self.glyphSet.charStrings[self.name].draw(pen, self.glyphSet.blender)


def _setCoordinates(glyph, coord, glyfTable):
    # Handle phantom points for (left, right, top, bottom) positions.
    assert len(coord) >= 4
    leftSideX = coord[-4][0]
    rightSideX = coord[-3][0]
    topSideY = coord[-2][1]
    bottomSideY = coord[-1][1]

    for _ in range(4):
        del coord[-1]

    if glyph.isComposite():
        assert len(coord) == len(glyph.components)
        glyph.components = [copy(comp) for comp in glyph.components]  # Shallow copy
        for p, comp in zip(coord, glyph.components):
            if hasattr(comp, "x"):
                comp.x, comp.y = p
    elif glyph.numberOfContours == 0:
        assert len(coord) == 0
    else:
        assert len(coord) == len(glyph.coordinates)
        glyph.coordinates = coord

    glyph.recalcBounds(glyfTable)

    horizontalAdvanceWidth = otRound(rightSideX - leftSideX)
    verticalAdvanceWidth = otRound(topSideY - bottomSideY)
    leftSideBearing = otRound(glyph.xMin - leftSideX)
    topSideBearing = otRound(topSideY - glyph.yMax)
    return (
        horizontalAdvanceWidth,
        leftSideBearing,
        verticalAdvanceWidth,
        topSideBearing,
    )
