# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import struct
from typing import Any, Callable, NamedTuple, Union

from PIL import Image
from PIL.TiffTags import TAGS_V2 as TIFF_TAGS

BytesLike = Union[bytes, memoryview]
MutableBytesLike = Union[bytearray, memoryview]


def _next_multiple(n: int, k: int) -> int:
    """Return the multiple of k that is greater than or equal n.

    >>> _next_multiple(101, 4)
    104
    >>> _next_multiple(100, 4)
    100
    """
    div, mod = divmod(n, k)
    if mod > 0:
        div += 1
    return div * k


def unpack_subbyte_pixels(
    packed: BytesLike, size: tuple[int, int], bits: int, scale: int = 0
) -> tuple[BytesLike, int]:
    """Unpack subbyte *bits* pixels into full bytes and rescale.

    When scale is 0, the appropriate scale is calculated.
    e.g. for 2-bit, the scale is adjusted so that
        0b00 = 0.00 = 0x00
        0b01 = 0.33 = 0x55
        0b10 = 0.66 = 0xaa
        0b11 = 1.00 = 0xff
    When scale is 1, no scaling is applied, appropriate when
    the bytes are palette indexes.
    """
    width, height = size
    bits_per_byte = 8 // bits
    stride = _next_multiple(width, bits_per_byte)
    buffer = bytearray(bits_per_byte * stride * height)
    max_read = len(buffer) // bits_per_byte
    if scale == 0:
        scale = 255 / ((2**bits) - 1)
    if bits == 4:
        _4bit_inner_loop(packed[:max_read], buffer, scale)
    elif bits == 2:
        _2bit_inner_loop(packed[:max_read], buffer, scale)
    # elif bits == 1:
    #     _1bit_inner_loop(packed[:max_read], buffer, scale)
    else:
        raise NotImplementedError(bits)
    return memoryview(buffer), stride


# def _1bit_inner_loop(in_: BytesLike, out: MutableBytesLike, scale: int) -> None:
#     """Unpack 1-bit values to their 8-bit equivalents.

#     Thus *out* must be 8x at long as *in*.
#     """
#     for n, val in enumerate(in_):
#         out[8 * n + 0] = int((val >> 7) & 0b1) * scale
#         out[8 * n + 1] = int((val >> 6) & 0b1) * scale
#         out[8 * n + 2] = int((val >> 5) & 0b1) * scale
#         out[8 * n + 3] = int((val >> 4) & 0b1) * scale
#         out[8 * n + 4] = int((val >> 3) & 0b1) * scale
#         out[8 * n + 5] = int((val >> 2) & 0b1) * scale
#         out[8 * n + 6] = int((val >> 1) & 0b1) * scale
#         out[8 * n + 7] = int((val >> 0) & 0b1) * scale


def _2bit_inner_loop(in_: BytesLike, out: MutableBytesLike, scale: int) -> None:
    """Unpack 2-bit values to their 8-bit equivalents.

    Thus *out* must be 4x at long as *in*.

    Images of this type are quite rare in practice, so we don't
    optimize this loop.
    """
    for n, val in enumerate(in_):
        out[4 * n] = int((val >> 6) * scale)
        out[4 * n + 1] = int(((val >> 4) & 0b11) * scale)
        out[4 * n + 2] = int(((val >> 2) & 0b11) * scale)
        out[4 * n + 3] = int((val & 0b11) * scale)


def _4bit_inner_loop(in_: BytesLike, out: MutableBytesLike, scale: int) -> None:
    """Unpack 4-bit values to their 8-bit equivalents.

    Thus *out* must be 2x at long as *in*.

    Images of this type are quite rare in practice, so we don't
    optimize this loop.
    """
    for n, val in enumerate(in_):
        out[2 * n] = int((val >> 4) * scale)
        out[2 * n + 1] = int((val & 0b1111) * scale)


def image_from_byte_buffer(buffer: BytesLike, size: tuple[int, int], stride: int):
    """Use Pillow to create one-component image from a byte buffer.

    *stride* is the number of bytes per row, and is essential for packed bits
    with odd image widths.
    """
    ystep = 1  # image is top to bottom in memory
    return Image.frombuffer('L', size, buffer, "raw", 'L', stride, ystep)


def _make_rgb_palette(gray_palette: bytes) -> bytes:
    palette = b''
    for entry in gray_palette:
        palette += bytes([entry]) * 3
    return palette


def _depalettize_cmyk(buffer: BytesLike, palette: BytesLike):
    with memoryview(buffer) as mv:
        output = bytearray(4 * len(mv))
        for n, pal_idx in enumerate(mv):
            output[4 * n : 4 * (n + 1)] = palette[4 * pal_idx : 4 * (pal_idx + 1)]
    return output


def image_from_buffer_and_palette(
    buffer: BytesLike,
    size: tuple[int, int],
    stride: int,
    base_mode: str,
    palette: BytesLike,
) -> Image.Image:
    """Construct an image from a byte buffer and apply the palette.

    1/2/4-bit images must be unpacked (no scaling!) to byte buffers first, such
    that every 8-bit integer is an index into the palette.
    """
    # Reminder Pillow palette byte order unintentionally changed in 8.3.0
    # https://github.com/python-pillow/Pillow/issues/5595
    # 8.2.0: all aligned by channel (very nonstandard)
    # 8.3.0: all channels for one color followed by the next color (e.g. RGBRGBRGB)

    if base_mode == 'RGB':
        im = image_from_byte_buffer(buffer, size, stride)
        im.putpalette(palette, rawmode=base_mode)
    elif base_mode == 'L':
        # Pillow does not fully support palettes with rawmode='L'.
        # Convert to RGB palette.
        gray_palette = _make_rgb_palette(palette)
        im = image_from_byte_buffer(buffer, size, stride)
        im.putpalette(gray_palette, rawmode='RGB')
    elif base_mode == 'CMYK':
        # Pillow does not support CMYK with palettes; convert manually
        output = _depalettize_cmyk(buffer, palette)
        im = Image.frombuffer('CMYK', size, data=output, decoder_name='raw')
    else:
        raise NotImplementedError(f'palette with {base_mode}')
    return im


def fix_1bit_palette_image(
    im: Image.Image, base_mode: str, palette: BytesLike
) -> Image.Image:
    """Apply palettes to 1-bit images."""
    im = im.convert('P')
    if base_mode == 'RGB' and len(palette) == 6:
        # rgbrgb -> rgb000000...rgb
        expanded_palette = b''.join(
            [palette[0:3], (b'\x00\x00\x00' * (256 - 2)), palette[3:6]]
        )
        im.putpalette(expanded_palette, rawmode='RGB')
    elif base_mode == 'L':
        try:
            im.putpalette(palette, rawmode='L')
        except ValueError as e:
            if 'unrecognized raw mode' in str(e):
                rgb_palette = _make_rgb_palette(palette)
                im.putpalette(rgb_palette, rawmode='RGB')
    return im


def generate_ccitt_header(
    size: tuple[int, int],
    *,
    data_length: int,
    ccitt_group: int,
    t4_options: int | None,
    photometry: int,
    icc: bytes,
) -> bytes:
    """Generate binary CCITT header for image with given parameters."""
    tiff_header_struct = '<' + '2s' + 'H' + 'L' + 'H'

    tag_keys = {tag.name: key for key, tag in TIFF_TAGS.items()}  # type: ignore
    ifd_struct = '<HHLL'

    class IFD(NamedTuple):
        key: int
        typecode: Any
        count_: int
        data: int | Callable[[], int | None]

    ifds: list[IFD] = []

    def header_length(ifd_count) -> int:
        return (
            struct.calcsize(tiff_header_struct)
            + struct.calcsize(ifd_struct) * ifd_count
            + 4
        )

    def add_ifd(tag_name: str, data: int | Callable[[], int | None], count: int = 1):
        key = tag_keys[tag_name]
        typecode = TIFF_TAGS[key].type  # type: ignore
        ifds.append(IFD(key, typecode, count, data))

    image_offset = None
    width, height = size
    add_ifd('ImageWidth', width)
    add_ifd('ImageLength', height)
    add_ifd('BitsPerSample', 1)
    add_ifd('Compression', ccitt_group)
    add_ifd('FillOrder', 1)
    if t4_options is not None:
        add_ifd('T4Options', t4_options)
    add_ifd('PhotometricInterpretation', photometry)
    add_ifd('StripOffsets', lambda: image_offset)
    add_ifd('RowsPerStrip', height)
    add_ifd('StripByteCounts', data_length)

    icc_offset = 0
    if icc:
        add_ifd('ICCProfile', lambda: icc_offset, count=len(icc))

    icc_offset = header_length(len(ifds))
    image_offset = icc_offset + len(icc)

    ifd_args = [(arg() if callable(arg) else arg) for ifd in ifds for arg in ifd]
    tiff_header = struct.pack(
        (tiff_header_struct + ifd_struct[1:] * len(ifds) + 'L'),
        b'II',  # Byte order indication: Little endian
        42,  # Version number (always 42)
        8,  # Offset to first IFD
        len(ifds),  # Number of tags in IFD
        *ifd_args,
        0,  # Last IFD
    )

    if icc:
        tiff_header += icc
    return tiff_header
