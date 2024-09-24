# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Implement pdfdoc codec."""

from __future__ import annotations

import codecs
from typing import Any, Container

from pikepdf._core import pdf_doc_to_utf8, utf8_to_pdf_doc

# pylint: disable=redefined-builtin

# See PDF Reference Manual 1.7, Table D.2.
# The following generates set of all Unicode code points that can be encoded in
# pdfdoc. Since pdfdoc is 8-bit, the vast majority of code points cannot be.

# Due to a bug, qpdf <= 10.5 and pikepdf < 5 had some inconsistencies around
# PdfDocEncoding.
PDFDOC_ENCODABLE = frozenset(
    list(range(0x00, 0x17 + 1))
    + list(range(0x20, 0x7E + 1))
    + [
        0x2022,
        0x2020,
        0x2021,
        0x2026,
        0x2014,
        0x2013,
        0x0192,
        0x2044,
        0x2039,
        0x203A,
        0x2212,
        0x2030,
        0x201E,
        0x201C,
        0x201D,
        0x2018,
        0x2019,
        0x201A,
        0x2122,
        0xFB01,
        0xFB02,
        0x0141,
        0x0152,
        0x0160,
        0x0178,
        0x017D,
        0x0131,
        0x0142,
        0x0153,
        0x0161,
        0x017E,
        0x20AC,
    ]
    + [0x02D8, 0x02C7, 0x02C6, 0x02D9, 0x02DD, 0x02DB, 0x02DA, 0x02DC]
    + list(range(0xA1, 0xAC + 1))
    + list(range(0xAE, 0xFF + 1))
)


def _find_first_index(s: str, ordinals: Container[int]) -> int:
    for n, char in enumerate(s):
        if ord(char) not in ordinals:
            return n
    raise ValueError("couldn't find the unencodable character")  # pragma: no cover


def pdfdoc_encode(input: str, errors: str = 'strict') -> tuple[bytes, int]:
    """Convert input string to bytes in PdfDocEncoding."""
    error_marker = b'?' if errors == 'replace' else b'\xad'
    success, pdfdoc = utf8_to_pdf_doc(input, error_marker)
    if success:
        return pdfdoc, len(input)

    if errors == 'ignore':
        pdfdoc = pdfdoc.replace(b'\xad', b'')
        return pdfdoc, len(input)
    if errors == 'replace':
        return pdfdoc, len(input)
    if errors == 'strict':
        if input.startswith('\xfe\xff') or input.startswith('\xff\xfe'):
            raise UnicodeEncodeError(
                'pdfdoc',
                input,
                0,
                2,
                "strings beginning with byte order marks cannot be encoded in pdfdoc",
            )

        # libqpdf doesn't return what character caused the error, and Python
        # needs this, so make an educated guess and raise an exception based
        # on that.
        offending_index = _find_first_index(input, PDFDOC_ENCODABLE)
        raise UnicodeEncodeError(
            'pdfdoc',
            input,
            offending_index,
            offending_index + 1,
            "character cannot be represented in pdfdoc encoding",
        )
    raise LookupError(errors)


def pdfdoc_decode(input: bytes, errors: str = 'strict') -> tuple[str, int]:
    """Convert PdfDoc-encoded input into a Python str."""
    if isinstance(input, memoryview):
        input = input.tobytes()
    s = pdf_doc_to_utf8(input)
    if errors == 'strict':
        idx = s.find('\ufffd')
        if idx >= 0:
            raise UnicodeDecodeError(
                'pdfdoc',
                input,
                idx,
                idx + 1,
                "no Unicode mapping is defined for this character",
            )

    return s, len(input)


class PdfDocCodec(codecs.Codec):
    """Implement PdfDocEncoding character map used inside PDFs."""

    def encode(self, input: str, errors: str = 'strict') -> tuple[bytes, int]:
        """Implement codecs.Codec.encode for pdfdoc."""
        return pdfdoc_encode(input, errors)

    def decode(self, input: bytes, errors: str = 'strict') -> tuple[str, int]:
        """Implement codecs.Codec.decode for pdfdoc."""
        return pdfdoc_decode(input, errors)


class PdfDocStreamWriter(PdfDocCodec, codecs.StreamWriter):
    """Implement PdfDocEncoding stream writer."""


class PdfDocStreamReader(PdfDocCodec, codecs.StreamReader):
    """Implement PdfDocEncoding stream reader."""

    def decode(self, input: bytes, errors: str = 'strict') -> tuple[str, int]:
        """Implement codecs.StreamReader.decode for pdfdoc."""
        return PdfDocCodec.decode(self, input, errors)


class PdfDocIncrementalEncoder(codecs.IncrementalEncoder):
    """Implement PdfDocEncoding incremental encoder."""

    def encode(self, input: str, final: bool = False) -> bytes:
        """Implement codecs.IncrementalEncoder.encode for pdfdoc."""
        return pdfdoc_encode(input, 'strict')[0]


class PdfDocIncrementalDecoder(codecs.IncrementalDecoder):
    """Implement PdfDocEncoding incremental decoder."""

    def decode(self, input: Any, final: bool = False) -> str:  # type: ignore
        """Implement codecs.IncrementalDecoder.decode for pdfdoc."""
        return pdfdoc_decode(bytes(input), 'strict')[0]


def find_pdfdoc(encoding: str) -> codecs.CodecInfo | None:
    """Register pdfdoc codec with Python.

    Both pdfdoc and pdfdoc_pikepdf are registered. Use "pdfdoc_pikepdf" if pikepdf's
    codec is required. If another third party package installs a codec named pdfdoc,
    the first imported by Python will be registered and will service all encoding.
    Unfortunately, Python's codec infrastructure does not give a better mechanism
    for resolving conflicts.
    """
    if encoding in ('pdfdoc', 'pdfdoc_pikepdf'):
        codec = PdfDocCodec()
        return codecs.CodecInfo(
            name=encoding,
            encode=codec.encode,
            decode=codec.decode,
            streamwriter=PdfDocStreamWriter,
            streamreader=PdfDocStreamReader,
            incrementalencoder=PdfDocIncrementalEncoder,
            incrementaldecoder=PdfDocIncrementalDecoder,
        )
    return None  # pragma: no cover


codecs.register(find_pdfdoc)

__all__ = ['utf8_to_pdf_doc', 'pdf_doc_to_utf8']
