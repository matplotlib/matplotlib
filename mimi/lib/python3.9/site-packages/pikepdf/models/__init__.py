# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Python implementation of higher level PDF constructs."""

from __future__ import annotations

from ._content_stream import (
    ContentStreamInstructions,
    PdfParsingError,
    UnparseableContentStreamInstructions,
    parse_content_stream,
    unparse_content_stream,
)
from .encryption import Encryption, EncryptionInfo, Permissions
from .image import PdfImage, PdfInlineImage, UnsupportedImageTypeError
from .matrix import PdfMatrix
from .metadata import PdfMetadata
from .outlines import (
    Outline,
    OutlineItem,
    OutlineStructureError,
    PageLocation,
    make_page_destination,
)

__all__ = [
    'ContentStreamInstructions',
    'PdfParsingError',
    'UnparseableContentStreamInstructions',
    'parse_content_stream',
    'unparse_content_stream',
    'Encryption',
    'EncryptionInfo',
    'Permissions',
    'PdfImage',
    'PdfInlineImage',
    'UnsupportedImageTypeError',
    'PdfMatrix',
    'PdfMetadata',
    'Outline',
    'OutlineItem',
    'OutlineStructureError',
    'PageLocation',
    'make_page_destination',
]
