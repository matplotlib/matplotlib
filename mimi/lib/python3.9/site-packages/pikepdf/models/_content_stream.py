# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Content stream parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Collection, List, Tuple, Union, cast

from pikepdf import Object, ObjectType, Operator, Page, PdfError, _core

if TYPE_CHECKING:
    from pikepdf.models.image import PdfInlineImage

# Operands, Operator
_OldContentStreamOperands = Collection[Union[Object, 'PdfInlineImage']]
_OldContentStreamInstructions = Tuple[_OldContentStreamOperands, Operator]

ContentStreamInstructions = Union[
    _core.ContentStreamInstruction, _core.ContentStreamInlineImage
]

UnparseableContentStreamInstructions = Union[
    ContentStreamInstructions, _OldContentStreamInstructions
]


class PdfParsingError(Exception):
    """Error when parsing a PDF content stream."""

    def __init__(self, message=None, line=None):
        if not message:
            message = f"Error encoding content stream at line {line}"
        super().__init__(message)
        self.line = line


def parse_content_stream(
    page_or_stream: Object | Page, operators: str = ''
) -> list[ContentStreamInstructions]:
    """Parse a PDF content stream into a sequence of instructions.

    A PDF content stream is list of instructions that describe where to render
    the text and graphics in a PDF. This is the starting point for analyzing
    PDFs.

    If the input is a page and page.Contents is an array, then the content
    stream is automatically treated as one coalesced stream.

    Each instruction contains at least one operator and zero or more operands.

    This function does not have anything to do with opening a PDF file itself or
    processing data from a whole PDF. It is for processing a specific object inside
    a PDF that is already opened.

    Args:
        page_or_stream: A page object, or the content
            stream attached to another object such as a Form XObject.
        operators: A space-separated string of operators to whitelist.
            For example 'q Q cm Do' will return only operators
            that pertain to drawing images. Use 'BI ID EI' for inline images.
            All other operators and associated tokens are ignored. If blank,
            all tokens are accepted.

    Example:
        >>> with pikepdf.Pdf.open(input_pdf) as pdf:
        >>>     page = pdf.pages[0]
        >>>     for operands, command in parse_content_stream(page):
        >>>         print(command)

    .. versionchanged:: 3.0
        Returns a list of ``ContentStreamInstructions`` instead of a list
        of (operand, operator) tuples. The returned items are duck-type compatible
        with the previous returned items.
    """
    if not isinstance(page_or_stream, (Object, Page)):
        raise TypeError("stream must be a pikepdf.Object or pikepdf.Page")

    if (
        isinstance(page_or_stream, Object)
        and page_or_stream._type_code != ObjectType.stream
        and page_or_stream.get('/Type') != '/Page'
    ):
        raise TypeError("parse_content_stream called on page or stream object")

    if isinstance(page_or_stream, Page):
        page_or_stream = page_or_stream.obj

    try:
        if page_or_stream.get('/Type') == '/Page':
            page = page_or_stream
            instructions = cast(
                List[ContentStreamInstructions],
                page._parse_page_contents_grouped(operators),
            )
        else:
            stream = page_or_stream
            instructions = cast(
                List[ContentStreamInstructions],
                Object._parse_stream_grouped(stream, operators),
            )
    except PdfError as e:
        if 'supposed to be a stream or an array' in str(e):
            raise TypeError("parse_content_stream called on non-stream Object") from e
        raise e from e

    return instructions


def unparse_content_stream(
    instructions: Collection[UnparseableContentStreamInstructions],
) -> bytes:
    """Convert collection of instructions to bytes suitable for storing in PDF.

    Given a parsed list of instructions/operand-operators, convert to bytes suitable
    for embedding in a PDF. In PDF the operator always follows the operands.

    Args:
        instructions: collection of instructions such as is returned
            by :func:`parse_content_stream()`

    Returns:
        A binary content stream, suitable for attaching to a Pdf.
        To attach to a Pdf, use :meth:`Pdf.make_stream()``.

    .. versionchanged:: 3.0
        Now accept collections that contain any mixture of
        ``ContentStreamInstruction``, ``ContentStreamInlineImage``, and the older
        operand-operator tuples from pikepdf 2.x.
    """
    try:
        return _core._unparse_content_stream(instructions)
    except (ValueError, TypeError, RuntimeError) as e:
        raise PdfParsingError(
            "While unparsing a content stream, an error occurred"
        ) from e
