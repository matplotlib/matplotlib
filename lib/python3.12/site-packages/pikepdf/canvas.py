# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Module for generating PDF content streams."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path

from PIL import Image

from pikepdf import (
    Array,
    ContentStreamInstruction,
    Dictionary,
    Matrix,
    Name,
    Operator,
    Pdf,
    unparse_content_stream,
)
from pikepdf.objects import String

log = logging.getLogger(__name__)


Color = namedtuple('Color', ['red', 'green', 'blue', 'alpha'])

BLACK = Color(0, 0, 0, 1)
WHITE = Color(1, 1, 1, 1)
BLUE = Color(0, 0, 1, 1)
CYAN = Color(0, 1, 1, 1)
GREEN = Color(0, 1, 0, 1)
DARKGREEN = Color(0, 0.5, 0, 1)
MAGENTA = Color(1, 0, 1, 1)
RED = Color(1, 0, 0, 1)


class TextDirection(Enum):
    """Enumeration for text direction."""

    LTR = 1  # Left to right: the default
    RTL = 2  # Right to left: Arabic, Hebrew, Persian


class Font(ABC):
    """Base class for fonts."""

    @abstractmethod
    def text_width(self, text: str, fontsize: float) -> float:
        """Estimate the width of a text string when rendered with the given font."""

    @abstractmethod
    def register(self, pdf: Pdf) -> Dictionary:
        """Register the font.

        Create several data structures in the Pdf to describe the font. While it create
        the data, a reference should be set in at least one page's /Resources dictionary
        to retain the font in the output PDF and ensure it is usable on that page.

        The returned Dictionary should be created as an indirect object, using
        ``pdf.make_indirect()``.

        Returns a Dictionary suitable for insertion into a /Resources /Font dictionary.
        """


class Helvetica(Font):
    """Helvetica font."""

    def text_width(self, text: str, fontsize: float) -> float:
        """Estimate the width of a text string when rendered with the given font."""
        raise NotImplementedError()

    def register(self, pdf: Pdf) -> Dictionary:
        """Register the font."""
        return pdf.make_indirect(
            Dictionary(
                BaseFont=Name.Helvetica,
                Type=Name.Font,
                Subtype=Name.Type1,
            )
        )


class ContentStreamBuilder:
    """Content stream builder."""

    def __init__(self):
        """Initialize."""
        self._stream = b""

    def _append(self, inst: ContentStreamInstruction):
        self._stream += unparse_content_stream([inst]) + b"\n"

    def extend(self, other: ContentStreamBuilder):
        """Append another content stream."""
        self._stream += other._stream

    def push(self):
        """Save the graphics state."""
        inst = ContentStreamInstruction([], Operator("q"))
        self._append(inst)
        return self

    def pop(self):
        """Restore the graphics state."""
        inst = ContentStreamInstruction([], Operator("Q"))
        self._append(inst)
        return self

    def cm(self, matrix: Matrix):
        """Concatenate matrix."""
        inst = ContentStreamInstruction(matrix.shorthand, Operator("cm"))
        self._append(inst)
        return self

    def begin_text(self):
        """Begin text object."""
        inst = ContentStreamInstruction([], Operator("BT"))
        self._append(inst)
        return self

    def end_text(self):
        """End text object."""
        inst = ContentStreamInstruction([], Operator("ET"))
        self._append(inst)
        return self

    def begin_marked_content_proplist(self, mctype: Name, mcid: int):
        """Begin marked content sequence."""
        inst = ContentStreamInstruction(
            [mctype, Dictionary(MCID=mcid)], Operator("BDC")
        )
        self._append(inst)
        return self

    def begin_marked_content(self, mctype: Name):
        """Begin marked content sequence."""
        inst = ContentStreamInstruction([mctype], Operator("BMC"))
        self._append(inst)
        return self

    def end_marked_content(self):
        """End marked content sequence."""
        inst = ContentStreamInstruction([], Operator("EMC"))
        self._append(inst)
        return self

    def set_text_font(self, font: Name, size: int):
        """Set text font and size."""
        inst = ContentStreamInstruction([font, size], Operator("Tf"))
        self._append(inst)
        return self

    def set_text_matrix(self, matrix: Matrix):
        """Set text matrix."""
        inst = ContentStreamInstruction(matrix.shorthand, Operator("Tm"))
        self._append(inst)
        return self

    def set_text_rendering(self, mode: int):
        """Set text rendering mode."""
        inst = ContentStreamInstruction([mode], Operator("Tr"))
        self._append(inst)
        return self

    def set_text_horizontal_scaling(self, scale: float):
        """Set text horizontal scaling."""
        inst = ContentStreamInstruction([scale], Operator("Tz"))
        self._append(inst)
        return self

    def show_text(self, encoded: bytes):
        """Show text.

        The text must be encoded in character codes expected by the font.
        """
        # [ <text string> ] TJ
        # operands need to be enclosed in Array
        inst = ContentStreamInstruction([Array([String(encoded)])], Operator("TJ"))
        self._append(inst)
        return self

    def move_cursor(self, dx, dy):
        """Move cursor."""
        inst = ContentStreamInstruction([dx, dy], Operator("Td"))
        self._append(inst)
        return self

    def stroke_and_close(self):
        """Stroke and close path."""
        inst = ContentStreamInstruction([], Operator("s"))
        self._append(inst)
        return self

    def fill(self):
        """Stroke and close path."""
        inst = ContentStreamInstruction([], Operator("f"))
        self._append(inst)
        return self

    def append_rectangle(self, x: float, y: float, w: float, h: float):
        """Append rectangle to path."""
        inst = ContentStreamInstruction([x, y, w, h], Operator("re"))
        self._append(inst)
        return self

    def set_stroke_color(self, r: float, g: float, b: float):
        """Set RGB stroke color."""
        inst = ContentStreamInstruction([r, g, b], Operator("RG"))
        self._append(inst)
        return self

    def set_fill_color(self, r: float, g: float, b: float):
        """Set RGB fill color."""
        inst = ContentStreamInstruction([r, g, b], Operator("rg"))
        self._append(inst)
        return self

    def set_line_width(self, width):
        """Set line width."""
        inst = ContentStreamInstruction([width], Operator("w"))
        self._append(inst)
        return self

    def line(self, x1: float, y1: float, x2: float, y2: float):
        """Draw line."""
        insts = [
            ContentStreamInstruction([x1, y1], Operator("m")),
            ContentStreamInstruction([x2, y2], Operator("l")),
        ]
        self._append(insts[0])
        self._append(insts[1])
        return self

    def set_dashes(self, array=None, phase=0):
        """Set dashes."""
        if array is None:
            array = []
        if isinstance(array, (int, float)):
            array = (array, phase)
            phase = 0
        inst = ContentStreamInstruction([array, phase], Operator("d"))
        self._append(inst)
        return self

    def draw_xobject(self, name: Name):
        """Draw XObject.

        Add instructions to render an XObject. The XObject must be
        defined in the document.

        Args:
            name: Name of XObject
        """
        inst = ContentStreamInstruction([name], Operator("Do"))
        self._append(inst)
        return self

    def build(self) -> bytes:
        """Build content stream."""
        return self._stream


@dataclass
class LoadedImage:
    """Loaded image."""

    name: Name
    image: Image.Image


class _CanvasAccessor:
    """Contains all drawing methods class for drawing on a Canvas."""

    def __init__(self, cs: ContentStreamBuilder, images=None):
        self._cs = cs
        self._images = images if images is not None else []
        self._stack_depth = 0

    def stroke_color(self, color: Color):
        """Set stroke color."""
        r, g, b = color.red, color.green, color.blue
        self._cs.set_stroke_color(r, g, b)
        return self

    def fill_color(self, color: Color):
        """Set fill color."""
        r, g, b = color.red, color.green, color.blue
        self._cs.set_fill_color(r, g, b)
        return self

    def line_width(self, width):
        """Set line width."""
        self._cs.set_line_width(width)
        return self

    def line(self, x1, y1, x2, y2):
        """Draw line from (x1,y1) to (x2,y2)."""
        self._cs.line(x1, y1, x2, y2)
        self._cs.stroke_and_close()
        return self

    def rect(self, x, y, w, h, fill: bool):
        """Draw optionally filled rectangle at (x,y) with width w and height h."""
        self._cs.append_rectangle(x, y, w, h)
        if fill:
            self._cs.fill()
        else:
            self._cs.stroke_and_close()
        return self

    def draw_image(self, image: Path | str | Image.Image, x, y, width, height):
        """Draw image at (x,y) with width w and height h."""
        with self.save_state(cm=Matrix(width, 0, 0, height, x, y)):
            if isinstance(image, (Path, str)):
                image = Image.open(image)
            image.load()
            if image.mode == "P":
                image = image.convert("RGB")
            if image.mode not in ("1", "L", "RGB"):
                raise ValueError(f"Unsupported image mode: {image.mode}")
            name = Name.random(prefix="Im")
            li = LoadedImage(name, image)
            self._images.append(li)
            self._cs.draw_xobject(name)
        return self

    def draw_text(self, text: Text):
        """Draw text object."""
        self._cs.extend(text._cs)
        self._cs.end_text()
        return self

    def dashes(self, *args):
        """Set dashes."""
        self._cs.set_dashes(*args)
        return self

    def push(self):
        """Save the graphics state."""
        self._cs.push()
        self._stack_depth += 1
        return self

    def pop(self):
        """Restore the previous graphics state."""
        self._cs.pop()
        self._stack_depth -= 1
        return self

    @contextmanager
    def save_state(self, *, cm: Matrix | None = None):
        """Save the graphics state and restore it on exit.

        Optionally, concatenate a transformation matrix. Implements
        the commonly used pattern of:
            q cm ... Q
        """
        self.push()
        if cm is not None:
            self.cm(cm)
        yield self
        self.pop()

    def cm(self, matrix: Matrix):
        """Concatenate a new transformation matrix to the current matrix."""
        self._cs.cm(matrix)
        return self


class Canvas:
    """Canvas for rendering PDFs with pikepdf.

    All drawing is done on a pikepdf canvas using the .do property.
    This interface manages the graphics state of the canvas.

    A Canvas can be exported as a single page Pdf using .to_pdf. This Pdf can
    then be merged into other PDFs or written to a file.
    """

    def __init__(self, *, page_size: tuple[int | float, int | float]):
        """Initialize a canvas."""
        self.page_size = page_size
        self._pdf = Pdf.new()
        self._page = self._pdf.add_blank_page(page_size=page_size)
        self._page.Resources = Dictionary(Font=Dictionary(), XObject=Dictionary())
        self._cs = ContentStreamBuilder()
        self._images: list[LoadedImage] = []
        self._accessor = _CanvasAccessor(self._cs, self._images)
        self.do.push()

    def add_font(self, resource_name: Name, font: Font):
        """Add a font to the page."""
        self._page.Resources.Font[resource_name] = font.register(self._pdf)

    @property
    def do(self) -> _CanvasAccessor:
        """Do operations on the current graphics state."""
        return self._accessor

    def _save_image(self, li: LoadedImage):
        return self._pdf.make_stream(
            li.image.tobytes(),
            Width=li.image.width,
            Height=li.image.height,
            ColorSpace=(
                Name.DeviceGray if li.image.mode in ("1", "L") else Name.DeviceRGB
            ),
            Type=Name.XObject,
            Subtype=Name.Image,
            BitsPerComponent=1 if li.image.mode == '1' else 8,
        )

    def to_pdf(self) -> Pdf:
        """Render the canvas as a single page PDF."""
        self.do.pop()
        if self._accessor._stack_depth != 0:
            log.warning(
                "Graphics state stack is not empty when page saved - "
                "rendering may be incorrect"
            )
        self._page.Contents = self._pdf.make_stream(self._cs.build())
        for li in self._images:
            self._page.Resources.XObject[li.name] = self._save_image(li)
        bio = BytesIO()
        self._pdf.save(bio)
        bio.seek(0)
        result = Pdf.open(bio)

        # Reset the graphics state to before we saved the page
        self.do.push()
        return result

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.to_pdf()._repr_mimebundle_(include, exclude)


class Text:
    """Text object for rendering text on a pikepdf canvas."""

    def __init__(self, direction=TextDirection.LTR):
        """Initialize."""
        self._cs = ContentStreamBuilder()
        self._cs.begin_text()
        self._direction = direction

    def font(self, font: Name, size: float):
        """Set font and size."""
        self._cs.set_text_font(font, size)
        return self

    def render_mode(self, mode):
        """Set text rendering mode."""
        self._cs.set_text_rendering(mode)
        return self

    def text_transform(self, matrix: Matrix):
        """Set text matrix."""
        self._cs.set_text_matrix(matrix)
        return self

    def show(self, text: str | bytes):
        """Show text.

        The text must be encoded in character codes expected by the font.
        If a text string is passed, it will be encoded as UTF-16BE.
        Text rendering will not work properly if the font's character
        codes are not consistent with UTF-16BE. This is a rudimentary
        interface. You've been warned.
        """
        if isinstance(text, str):
            encoded = b"\xfe\xff" + text.encode("utf-16be")
        else:
            encoded = text
        if self._direction == TextDirection.LTR:
            self._cs.show_text(encoded)
        else:
            self._cs.begin_marked_content(Name.ReversedChars)
            self._cs.show_text(encoded)
            self._cs.end_marked_content()
        return self

    def horiz_scale(self, scale):
        """Set text horizontal scaling."""
        self._cs.set_text_horizontal_scaling(scale)
        return self

    def move_cursor(self, x, y):
        """Move cursor."""
        self._cs.move_cursor(x, y)
        return self
