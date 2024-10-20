# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""A library for manipulating PDFs."""

# isort:skip_file

from __future__ import annotations

from pikepdf._version import __version__

try:
    from pikepdf import _core
except ImportError as _e:  # pragma: no cover
    _msg = "pikepdf's extension library failed to import"
    raise ImportError(_msg) from _e

from pikepdf._core import (
    AccessMode,
    Annotation,
    AttachedFileSpec,
    ContentStreamInlineImage,
    ContentStreamInstruction,
    DataDecodingError,
    DeletedObjectError,
    ForeignObjectError,
    Job,
    JobUsageError,
    Matrix,
    NameTree,
    NumberTree,
    ObjectHelper,
    ObjectStreamMode,
    Page,
    PasswordError,
    Pdf,
    PdfError,
    Rectangle,
    StreamDecodeLevel,
    Token,
    TokenFilter,
    TokenType,
)
from pikepdf._exceptions import DependencyError
from pikepdf.objects import (
    Array,
    Dictionary,
    Name,
    Object,
    ObjectType,
    Operator,
    Stream,
    String,
)
from pikepdf.models import (
    Encryption,
    Outline,
    OutlineItem,
    OutlineStructureError,
    PageLocation,
    PdfImage,
    PdfInlineImage,
    Permissions,
    UnsupportedImageTypeError,
    make_page_destination,
    parse_content_stream,
    unparse_content_stream,
)

# Importing these will monkeypatch classes defined in C++ and register a new
# pdfdoc codec
# While _cpphelpers is intended to be called from our C++ code only, explicitly
# importing helps introspection tools like PyInstaller figure out that the module
# is necessary.
from pikepdf import _cpphelpers, _methods, codec  # noqa: F401, F841
from pikepdf import settings

__libqpdf_version__: str = _core.qpdf_version()

# Provide pikepdf.{open, new} -> pikepdf.Pdf.{open, new}
open = Pdf.open  # pylint: disable=redefined-builtin
new = Pdf.new

# Exclude .open, .new here from to make sure from pikepdf import * does not clobber
# builtins.open()
# Exclude codec, objects, jbig2 because we import the interesting bits from them
# directly to here.
_exclude_from__all__ = {'open', 'new', 'codec', 'objects', 'jbig2'}

__all__ = [
    'AccessMode',
    'Annotation',
    'AttachedFileSpec',
    'ContentStreamInlineImage',
    'ContentStreamInstruction',
    'DataDecodingError',
    'DeletedObjectError',
    'DependencyError',
    'ForeignObjectError',
    'Job',
    'JobUsageError',
    'Matrix',
    'NameTree',
    'NumberTree',
    'ObjectHelper',
    'ObjectStreamMode',
    'Page',
    'PasswordError',
    'Pdf',
    'PdfError',
    'Rectangle',
    'StreamDecodeLevel',
    'Token',
    'TokenFilter',
    'TokenType',
    'Array',
    'Dictionary',
    'Name',
    'Object',
    'ObjectType',
    'Operator',
    'Stream',
    'String',
    'models',
    'Encryption',
    'Outline',
    'OutlineItem',
    'OutlineStructureError',
    'PageLocation',
    'PdfImage',
    'PdfInlineImage',
    'Permissions',
    'UnsupportedImageTypeError',
    'make_page_destination',
    'parse_content_stream',
    'unparse_content_stream',
    'settings',
    '__libqpdf_version__',
    '__version__',
]
