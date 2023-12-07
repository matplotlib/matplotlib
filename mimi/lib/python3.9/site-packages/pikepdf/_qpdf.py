# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

# isort: skip_file
# type: ignore

# This module is deprecated - use pikepdf._core instead, if you must
# Remove for pikepdf 9
from warnings import warn as _warn

from pikepdf._core import (
    AccessMode,
    Annotation,
    AttachedFile,
    AttachedFileSpec,
    Attachments,
    Buffer,
    ContentStreamInlineImage,
    ContentStreamInstruction,
    DataDecodingError,
    DeletedObjectError,
    EncryptionMethod,
    ForeignObjectError,
    Job,
    JobUsageError,
    NameTree,
    NumberTree,
    Object,
    ObjectHelper,
    ObjectStreamMode,
    ObjectType,
    Page,
    PageList,
    PasswordError,
    Pdf,
    PdfError,
    Rectangle,
    StreamDecodeLevel,
    StreamParser,
    Token,
    TokenFilter,
    TokenType,
    get_decimal_precision,
    pdf_doc_to_utf8,
    qpdf_version,
    set_decimal_precision,
    set_flate_compression_level,
    unparse,
    utf8_to_pdf_doc,
)

__all__ = [
    'AccessMode',
    'Annotation',
    'AttachedFile',
    'AttachedFileSpec',
    'Attachments',
    'Buffer',
    'ContentStreamInlineImage',
    'ContentStreamInstruction',
    'DataDecodingError',
    'DeletedObjectError',
    'EncryptionMethod',
    'ForeignObjectError',
    'Job',
    'JobUsageError',
    'NameTree',
    'NumberTree',
    'Object',
    'ObjectHelper',
    'ObjectStreamMode',
    'ObjectType',
    'Page',
    'PageList',
    'PasswordError',
    'Pdf',
    'PdfError',
    'Rectangle',
    'StreamDecodeLevel',
    'StreamParser',
    'Token',
    'TokenFilter',
    'TokenType',
    'get_decimal_precision',
    'pdf_doc_to_utf8',
    'qpdf_version',
    'set_decimal_precision',
    'set_flate_compression_level',
    'unparse',
    'utf8_to_pdf_doc',
]

_warn("pikepdf._qpdf is deprecated, use pikepdf._core instead.", DeprecationWarning)
