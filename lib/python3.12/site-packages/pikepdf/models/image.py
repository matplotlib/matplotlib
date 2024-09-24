# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Extract images embedded in PDF."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from copy import copy
from decimal import Decimal
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from shutil import copyfileobj
from typing import Any, BinaryIO, Callable, NamedTuple, TypeVar, Union, cast

from PIL import Image
from PIL.ImageCms import ImageCmsProfile

from pikepdf import jbig2
from pikepdf._core import Buffer, Pdf, PdfError, StreamDecodeLevel
from pikepdf._exceptions import DependencyError
from pikepdf.models import _transcoding
from pikepdf.objects import (
    Array,
    Dictionary,
    Name,
    Object,
    Stream,
    String,
)

T = TypeVar('T')

if sys.version_info >= (3, 9):
    RGBDecodeArray = tuple[float, float, float, float, float, float]
    GrayDecodeArray = tuple[float, float]
    CMYKDecodeArray = tuple[float, float, float, float, float, float, float, float]
    DecodeArray = Union[RGBDecodeArray, GrayDecodeArray, CMYKDecodeArray]
else:
    RGBDecodeArray = Any
    GrayDecodeArray = Any
    CMYKDecodeArray = Any
    DecodeArray = Any


class UnsupportedImageTypeError(Exception):
    """This image is formatted in a way pikepdf does not supported."""


class NotExtractableError(Exception):
    """Indicates that an image cannot be directly extracted."""


class HifiPrintImageNotTranscodableError(NotExtractableError):
    """Image contains high fidelity printing information and cannot be extracted."""


class InvalidPdfImageError(Exception):
    """This image is not valid according to the PDF 1.7 specification."""


def _array_str(value: Object | str | list):
    """Simplify pikepdf objects to array of str. Keep streams, dictionaries intact."""

    def _convert(item):
        if isinstance(item, (list, Array)):
            return [_convert(subitem) for subitem in item]
        if isinstance(item, (Stream, Dictionary, bytes, int)):
            return item
        if isinstance(item, (Name, str)):
            return str(item)
        if isinstance(item, (String)):
            return bytes(item)
        raise NotImplementedError(value)

    result = _convert(value)
    if not isinstance(result, list):
        result = [result]
    return result


def _ensure_list(value: list[Object] | Dictionary | Array | Object) -> list[Object]:
    """Ensure value is a list of pikepdf.Object, if it was not already.

    To support DecodeParms which can be present as either an array of dicts or a single
    dict. It's easier to convert to an array of one dict.
    """
    if isinstance(value, list):
        return value
    return list(value.wrap_in_array().as_list())


def _metadata_from_obj(
    obj: Dictionary | Stream, name: str, type_: Callable[[Any], T], default: T
) -> T | None:
    """Retrieve metadata from a dictionary or stream and wrangle types."""
    val = getattr(obj, name, default)
    try:
        return type_(val)
    except TypeError:
        if val is None:
            return None
    raise NotImplementedError('Metadata access for ' + name)


class PaletteData(NamedTuple):
    """Returns the color space and binary representation of the palette.

    ``base_colorspace`` is typically ``"RGB"`` or ``"L"`` (for grayscale).

    ``palette`` is typically 256 or 256*3=768 bytes, for grayscale and RGB color
    respectively, with each unit/triplet being the grayscale/RGB triplet values.
    """

    base_colorspace: str
    palette: bytes


class PdfImageBase(ABC):
    """Abstract base class for images."""

    SIMPLE_COLORSPACES = {'/DeviceRGB', '/DeviceGray', '/CalRGB', '/CalGray'}
    MAIN_COLORSPACES = SIMPLE_COLORSPACES | {'/DeviceCMYK', '/CalCMYK', '/ICCBased'}
    PRINT_COLORSPACES = {'/Separation', '/DeviceN'}

    @abstractmethod
    def _metadata(self, name: str, type_: Callable[[Any], T], default: T) -> T:
        """Get metadata for this image type."""

    @property
    def width(self) -> int:
        """Width of the image data in pixels."""
        return self._metadata('Width', int, 0)

    @property
    def height(self) -> int:
        """Height of the image data in pixels."""
        return self._metadata('Height', int, 0)

    @property
    def image_mask(self) -> bool:
        """Return ``True`` if this is an image mask."""
        return self._metadata('ImageMask', bool, False)

    @property
    def _bpc(self) -> int | None:
        """Bits per component for this image (low-level)."""
        return self._metadata('BitsPerComponent', int, 0)

    @property
    def _colorspaces(self):
        """Colorspace (low-level)."""
        return self._metadata('ColorSpace', _array_str, [])

    @property
    def filters(self):
        """List of names of the filters that we applied to encode this image."""
        return self._metadata('Filter', _array_str, [])

    @property
    def _decode_array(self) -> DecodeArray:
        """Extract the /Decode array."""
        decode: list = self._metadata('Decode', _ensure_list, [])
        if decode and len(decode) in (2, 6, 8):
            return cast(DecodeArray, tuple(float(value) for value in decode))

        if self.colorspace in ('/DeviceGray', '/CalGray'):
            return (0.0, 1.0)
        if self.colorspace == ('/DeviceRGB', '/CalRGB'):
            return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        if self.colorspace == '/DeviceCMYK':
            return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        if self.colorspace == '/ICCBased':
            if self._approx_mode_from_icc() == 'L':
                return (0.0, 1.0)
            if self._approx_mode_from_icc() == 'RGB':
                return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

        raise NotImplementedError(
            "Don't how to retrieve default /Decode array for image" + repr(self)
        )

    @property
    def decode_parms(self):
        """List of the /DecodeParms, arguments to filters."""
        return self._metadata('DecodeParms', _ensure_list, [])

    @property
    def colorspace(self) -> str | None:
        """PDF name of the colorspace that best describes this image."""
        if self.image_mask:
            return None  # Undefined for image masks
        if self._colorspaces:
            if self._colorspaces[0] in self.MAIN_COLORSPACES:
                return self._colorspaces[0]
            if self._colorspaces[0] == '/Indexed':
                subspace = self._colorspaces[1]
                if isinstance(subspace, str) and subspace in self.MAIN_COLORSPACES:
                    return subspace
                if isinstance(subspace, list) and subspace[0] in (
                    '/ICCBased',
                    '/DeviceN',
                    '/CalGray',
                    '/CalRGB',
                ):
                    return subspace[0]
            if self._colorspaces[0] == '/DeviceN':
                return '/DeviceN'

        raise NotImplementedError(
            "not sure how to get colorspace: " + repr(self._colorspaces)
        )

    @property
    def bits_per_component(self) -> int:
        """Bits per component of this image."""
        if self._bpc is None or self._bpc == 0:
            return 1 if self.image_mask else 8
        return self._bpc

    @property
    @abstractmethod
    def icc(self) -> ImageCmsProfile | None:
        """Return ICC profile for this image if one is defined."""

    @property
    def indexed(self) -> bool:
        """Check if the image has a defined color palette."""
        return '/Indexed' in self._colorspaces

    def _colorspace_has_name(self, name):
        try:
            cs = self._colorspaces
            if cs[0] == '/Indexed' and cs[1][0] == name:
                return True
            if cs[0] == name:
                return True
        except (IndexError, AttributeError, KeyError):
            pass
        return False

    @property
    def is_device_n(self) -> bool:
        """Check if image has a /DeviceN (complex printing) colorspace."""
        return self._colorspace_has_name('/DeviceN')

    @property
    def is_separation(self) -> bool:
        """Check if image has a /DeviceN (complex printing) colorspace."""
        return self._colorspace_has_name('/Separation')

    @property
    def size(self) -> tuple[int, int]:
        """Size of image as (width, height)."""
        return self.width, self.height

    def _approx_mode_from_icc(self):
        if self.indexed:
            icc_profile = self._colorspaces[1][1]
        else:
            icc_profile = self._colorspaces[1]
        icc_profile_nchannels = int(icc_profile['/N'])

        if icc_profile_nchannels == 1:
            return 'L'

        # Multiple channels, need to open the profile and look
        mode_from_xcolor_space = {'RGB ': 'RGB', 'CMYK': 'CMYK'}
        xcolor_space = self.icc.profile.xcolor_space
        return mode_from_xcolor_space.get(xcolor_space, '')

    @property
    def mode(self) -> str:
        """``PIL.Image.mode`` equivalent for this image, where possible.

        If an ICC profile is attached to the image, we still attempt to resolve a Pillow
        mode.
        """
        m = ''
        if self.is_device_n:
            m = 'DeviceN'
        elif self.is_separation:
            m = 'Separation'
        elif self.indexed:
            m = 'P'
        elif self.colorspace == '/DeviceGray' and self.bits_per_component == 1:
            m = '1'
        elif self.colorspace == '/DeviceGray' and self.bits_per_component > 1:
            m = 'L'
        elif self.colorspace == '/DeviceRGB':
            m = 'RGB'
        elif self.colorspace == '/DeviceCMYK':
            m = 'CMYK'
        elif self.colorspace == '/ICCBased':
            try:
                m = self._approx_mode_from_icc()
            except (ValueError, TypeError) as e:
                raise NotImplementedError(
                    "Not sure how to handle PDF image of this type"
                ) from e
        if m == '':
            raise NotImplementedError(
                "Not sure how to handle PDF image of this type"
            ) from None
        return m

    @property
    def filter_decodeparms(self):
        """Return normalized the Filter and DecodeParms data.

        PDF has a lot of possible data structures concerning /Filter and
        /DecodeParms. /Filter can be absent or a name or an array, /DecodeParms
        can be absent or a dictionary (if /Filter is a name) or an array (if
        /Filter is an array). When both are arrays the lengths match.

        Normalize this into:
        [(/FilterName, {/DecodeParmName: Value, ...}), ...]

        The order of /Filter matters as indicates the encoding/decoding sequence.
        """
        return list(zip_longest(self.filters, self.decode_parms, fillvalue={}))

    @property
    def palette(self) -> PaletteData | None:
        """Retrieve the color palette for this image if applicable."""
        if not self.indexed:
            return None
        try:
            _idx, base, _hival, lookup = self._colorspaces
        except ValueError as e:
            raise ValueError('Not sure how to interpret this palette') from e
        if self.icc or self.is_device_n or self.is_separation or isinstance(base, list):
            base = str(base[0])
        else:
            base = str(base)
        lookup = bytes(lookup)
        if base not in self.MAIN_COLORSPACES and base not in self.PRINT_COLORSPACES:
            raise NotImplementedError(f"not sure how to interpret this palette: {base}")
        if base in ('/DeviceRGB', '/CalRGB'):
            base = 'RGB'
        elif base in ('/DeviceGray', '/CalGray'):
            base = 'L'
        elif base == '/DeviceCMYK':
            base = 'CMYK'
        elif base == '/DeviceN':
            base = 'DeviceN'
        elif base == '/Separation':
            base = 'Separation'
        elif base == '/ICCBased':
            base = self._approx_mode_from_icc()
        else:
            raise NotImplementedError(f"not sure how to interpret this palette: {base}")
        return PaletteData(base, lookup)

    @abstractmethod
    def as_pil_image(self) -> Image.Image:
        """Convert this PDF image to a Python PIL (Pillow) image."""

    def _repr_png_(self) -> bytes:
        """Display hook for IPython/Jupyter."""
        b = BytesIO()
        with self.as_pil_image() as im:
            im.save(b, 'PNG')
            return b.getvalue()


class PdfImage(PdfImageBase):
    """Support class to provide a consistent API for manipulating PDF images.

    The data structure for images inside PDFs is irregular and complex,
    making it difficult to use without introducing errors for less
    typical cases. This class addresses these difficulties by providing a
    regular, Pythonic API similar in spirit (and convertible to) the Python
    Pillow imaging library.
    """

    obj: Stream
    _icc: ImageCmsProfile | None
    _pdf_source: Pdf | None

    def __new__(cls, obj: Stream):
        """Construct a PdfImage... or a PdfJpxImage if that is what we really are."""
        try:
            # Check if JPXDecode is called for and initialize as PdfJpxImage
            filters = _ensure_list(obj.Filter)
            if Name.JPXDecode in filters:
                return super().__new__(PdfJpxImage)
        except (AttributeError, KeyError):
            # __init__ will deal with any other errors
            pass
        return super().__new__(PdfImage)

    def __init__(self, obj: Stream):
        """Construct a PDF image from a Image XObject inside a PDF.

        ``pim = PdfImage(page.Resources.XObject['/ImageNN'])``

        Args:
            obj: an Image XObject
        """
        if isinstance(obj, Stream) and obj.stream_dict.get("/Subtype") != "/Image":
            raise TypeError("can't construct PdfImage from non-image")
        self.obj = obj
        self._icc = None

    def __eq__(self, other):
        if not isinstance(other, PdfImageBase):
            return NotImplemented
        return self.obj == other.obj

    @classmethod
    def _from_pil_image(cls, *, pdf, page, name, image):  # pragma: no cover
        """Insert a PIL image into a PDF (rudimentary).

        Args:
            pdf (pikepdf.Pdf): the PDF to attach the image to
            page (pikepdf.Object): the page to attach the image to
            name (str or pikepdf.Name): the name to set the image
            image (PIL.Image.Image): the image to insert
        """
        data = image.tobytes()

        imstream = Stream(pdf, data)
        imstream.Type = Name('/XObject')
        imstream.Subtype = Name('/Image')
        if image.mode == 'RGB':
            imstream.ColorSpace = Name('/DeviceRGB')
        elif image.mode in ('1', 'L'):
            imstream.ColorSpace = Name('/DeviceGray')
        imstream.BitsPerComponent = 1 if image.mode == '1' else 8
        imstream.Width = image.width
        imstream.Height = image.height

        page.Resources.XObject[name] = imstream

        return cls(imstream)

    def _metadata(self, name, type_, default):
        return _metadata_from_obj(self.obj, name, type_, default)

    @property
    def _iccstream(self):
        if self.colorspace == '/ICCBased':
            if not self.indexed:
                return self._colorspaces[1]
            assert isinstance(self._colorspaces[1], list)
            return self._colorspaces[1][1]
        raise NotImplementedError("Don't know how to find ICC stream for image")

    @property
    def icc(self) -> ImageCmsProfile | None:
        """If an ICC profile is attached, return a Pillow object that describe it.

        Most of the information may be found in ``icc.profile``.
        """
        if self.colorspace not in ('/ICCBased', '/Indexed'):
            return None
        if not self._icc:
            iccstream = self._iccstream
            iccbuffer = iccstream.get_stream_buffer()
            iccbytesio = BytesIO(iccbuffer)
            try:
                self._icc = ImageCmsProfile(iccbytesio)
            except OSError as e:
                if str(e) == 'cannot open profile from string':
                    # ICC profile is corrupt
                    raise UnsupportedImageTypeError(
                        "ICC profile corrupt or not readable"
                    ) from e
        return self._icc

    def _remove_simple_filters(self):
        """Remove simple lossless compression where it appears."""
        COMPLEX_FILTERS = {
            '/DCTDecode',
            '/JPXDecode',
            '/JBIG2Decode',
            '/CCITTFaxDecode',
        }
        indices = [n for n, filt in enumerate(self.filters) if filt in COMPLEX_FILTERS]
        if len(indices) > 1:
            raise NotImplementedError(
                f"Object {self.obj.objgen} has compound complex filters: "
                f"{self.filters}. We cannot decompress this."
            )
        if len(indices) == 0:
            # No complex filter indices, so all filters are simple - remove them all
            return self.obj.read_bytes(StreamDecodeLevel.specialized), []

        n = indices[0]
        if n == 0:
            # The only filter is complex, so return
            return self.obj.read_raw_bytes(), self.filters

        obj_copy = copy(self.obj)
        obj_copy.Filter = Array([Name(f) for f in self.filters[:n]])
        obj_copy.DecodeParms = Array(self.decode_parms[:n])
        return obj_copy.read_bytes(StreamDecodeLevel.specialized), self.filters[n:]

    def _extract_direct(self, *, stream: BinaryIO) -> str | None:
        """Attempt to extract the image directly to a usable image file.

        If there is no way to extract the image without decompressing or
        transcoding then raise an exception. The type and format of image
        generated will vary.

        Args:
            stream: Writable file stream to write data to, e.g. an open file
        """

        def normal_dct_rgb() -> bool:
            # Normal DCTDecode RGB images have the default value of
            # /ColorTransform 1 and are actually in YUV. Such a file can be
            # saved as a standard JPEG. RGB JPEGs without YUV conversion can't
            # be saved as JPEGs, and are probably bugs. Some software in the
            # wild actually produces RGB JPEGs in PDFs (probably a bug).
            DEFAULT_CT_RGB = 1
            ct = DEFAULT_CT_RGB
            if self.filter_decodeparms[0][1] is not None:
                ct = self.filter_decodeparms[0][1].get(
                    '/ColorTransform', DEFAULT_CT_RGB
                )
            return self.mode == 'RGB' and ct == DEFAULT_CT_RGB

        def normal_dct_cmyk() -> bool:
            # Normal DCTDecode CMYKs have /ColorTransform 0 and can be saved.
            # There is a YUVK colorspace but CMYK JPEGs don't generally use it
            DEFAULT_CT_CMYK = 0
            ct = DEFAULT_CT_CMYK
            if self.filter_decodeparms[0][1] is not None:
                ct = self.filter_decodeparms[0][1].get(
                    '/ColorTransform', DEFAULT_CT_CMYK
                )
            return self.mode == 'CMYK' and ct == DEFAULT_CT_CMYK

        data, filters = self._remove_simple_filters()

        if filters == ['/CCITTFaxDecode']:
            if self.colorspace == '/ICCBased':
                icc = self._iccstream.read_bytes()
            else:
                icc = None
            stream.write(self._generate_ccitt_header(data, icc=icc))
            stream.write(data)
            return '.tif'
        if filters == ['/DCTDecode'] and (
            self.mode == 'L' or normal_dct_rgb() or normal_dct_cmyk()
        ):
            stream.write(data)
            return '.jpg'

        return None

    def _extract_transcoded_1248bits(self) -> Image.Image:
        """Extract an image when there are 1/2/4/8 bits packed in byte data."""
        stride = 0  # tell Pillow to calculate stride from line width
        scale = 0 if self.mode == 'L' else 1
        if self.bits_per_component in (2, 4):
            buffer, stride = _transcoding.unpack_subbyte_pixels(
                self.read_bytes(), self.size, self.bits_per_component, scale
            )
        elif self.bits_per_component == 8:
            buffer = cast(memoryview, self.get_stream_buffer())
        else:
            raise InvalidPdfImageError("BitsPerComponent must be 1, 2, 4, 8, or 16")

        if self.mode == 'P' and self.palette is not None:
            base_mode, palette = self.palette
            im = _transcoding.image_from_buffer_and_palette(
                buffer,
                self.size,
                stride,
                base_mode,
                palette,
            )
        else:
            im = _transcoding.image_from_byte_buffer(buffer, self.size, stride)
        return im

    def _extract_transcoded_1bit(self) -> Image.Image:
        if not self.image_mask and self.mode in ('RGB', 'CMYK'):
            raise UnsupportedImageTypeError("1-bit RGB and CMYK are not supported")
        try:
            data = self.read_bytes()
        except (RuntimeError, PdfError) as e:
            if (
                'read_bytes called on unfilterable stream' in str(e)
                and not jbig2.get_decoder().available()
            ):
                raise DependencyError(
                    "jbig2dec - not installed or installed version is too old "
                    "(older than version 0.15)"
                ) from None
            raise

        im = Image.frombytes('1', self.size, data)

        if self.palette is not None:
            base_mode, palette = self.palette
            im = _transcoding.fix_1bit_palette_image(im, base_mode, palette)

        return im

    def _extract_transcoded_mask(self) -> Image.Image:
        return self._extract_transcoded_1bit()

    def _extract_transcoded(self) -> Image.Image:
        if self.image_mask:
            return self._extract_transcoded_mask()

        if self.mode in {'DeviceN', 'Separation'}:
            raise HifiPrintImageNotTranscodableError()

        if self.mode == 'RGB' and self.bits_per_component == 8:
            # Cannot use the zero-copy .get_stream_buffer here, we have 3-byte
            # RGB and Pillow needs RGBX.
            im = Image.frombuffer(
                'RGB', self.size, self.read_bytes(), 'raw', 'RGB', 0, 1
            )
        elif self.mode == 'CMYK' and self.bits_per_component == 8:
            im = Image.frombuffer(
                'CMYK', self.size, self.get_stream_buffer(), 'raw', 'CMYK', 0, 1
            )
        # elif self.mode == '1':
        elif self.bits_per_component == 1:
            im = self._extract_transcoded_1bit()
        elif self.mode in ('L', 'P') and self.bits_per_component <= 8:
            im = self._extract_transcoded_1248bits()
        else:
            raise UnsupportedImageTypeError(repr(self) + ", " + repr(self.obj))

        if self.colorspace == '/ICCBased' and self.icc is not None:
            im.info['icc_profile'] = self.icc.tobytes()

        return im

    def _extract_to_stream(self, *, stream: BinaryIO) -> str:
        """Extract the image to a stream.

        If possible, the compressed data is extracted and inserted into
        a compressed image file format without transcoding the compressed
        content. If this is not possible, the data will be decompressed
        and extracted to an appropriate format.

        Args:
            stream: Writable stream to write data to

        Returns:
            The file format extension.
        """
        direct_extraction = self._extract_direct(stream=stream)
        if direct_extraction:
            return direct_extraction

        im = None
        try:
            im = self._extract_transcoded()
            if im.mode == 'CMYK':
                im.save(stream, format='tiff', compression='tiff_adobe_deflate')
                return '.tiff'
            if im:
                im.save(stream, format='png')
                return '.png'
        except PdfError as e:
            if 'called on unfilterable stream' in str(e):
                raise UnsupportedImageTypeError(repr(self)) from e
            raise
        finally:
            if im:
                im.close()

        raise UnsupportedImageTypeError(repr(self))

    def extract_to(
        self, *, stream: BinaryIO | None = None, fileprefix: str = ''
    ) -> str:
        """Extract the image directly to a usable image file.

        If possible, the compressed data is extracted and inserted into
        a compressed image file format without transcoding the compressed
        content. If this is not possible, the data will be decompressed
        and extracted to an appropriate format.

        Because it is not known until attempted what image format will be
        extracted, users should not assume what format they are getting back.
        When saving the image to a file, use a temporary filename, and then
        rename the file to its final name based on the returned file extension.

        Images might be saved as any of .png, .jpg, or .tiff.

        Examples:
            >>> im.extract_to(stream=bytes_io)  # doctest: +SKIP
            '.png'

            >>> im.extract_to(fileprefix='/tmp/image00')  # doctest: +SKIP
            '/tmp/image00.jpg'

        Args:
            stream: Writable stream to write data to.
            fileprefix (str or Path): The path to write the extracted image to,
                without the file extension.

        Returns:
            If *fileprefix* was provided, then the fileprefix with the
            appropriate extension. If no *fileprefix*, then an extension
            indicating the file type.
        """
        if bool(stream) == bool(fileprefix):
            raise ValueError("Cannot set both stream and fileprefix")
        if stream:
            return self._extract_to_stream(stream=stream)

        bio = BytesIO()
        extension = self._extract_to_stream(stream=bio)
        bio.seek(0)
        filepath = Path(str(Path(fileprefix)) + extension)
        with filepath.open('wb') as target:
            copyfileobj(bio, target)
        return str(filepath)

    def read_bytes(
        self, decode_level: StreamDecodeLevel = StreamDecodeLevel.specialized
    ) -> bytes:
        """Decompress this image and return it as unencoded bytes."""
        return self.obj.read_bytes(decode_level=decode_level)

    def get_stream_buffer(
        self, decode_level: StreamDecodeLevel = StreamDecodeLevel.specialized
    ) -> Buffer:
        """Access this image with the buffer protocol."""
        return self.obj.get_stream_buffer(decode_level=decode_level)

    def as_pil_image(self) -> Image.Image:
        """Extract the image as a Pillow Image, using decompression as necessary.

        Caller must close the image.
        """
        bio = BytesIO()
        direct_extraction = self._extract_direct(stream=bio)
        if direct_extraction:
            bio.seek(0)
            return Image.open(bio)

        im = self._extract_transcoded()
        if not im:
            raise UnsupportedImageTypeError(repr(self))

        return im

    def _generate_ccitt_header(self, data: bytes, icc: bytes | None = None) -> bytes:
        """Construct a CCITT G3 or G4 header from the PDF metadata."""
        # https://stackoverflow.com/questions/2641770/
        # https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf

        if not self.decode_parms:
            raise ValueError("/CCITTFaxDecode without /DecodeParms")

        expected_defaults = [
            ("/EncodedByteAlign", False),
        ]
        for name, val in expected_defaults:
            if self.decode_parms[0].get(name, val) != val:
                raise UnsupportedImageTypeError(
                    f"/CCITTFaxDecode with decode parameter {name} not equal {val}"
                )

        k = self.decode_parms[0].get("/K", 0)
        t4_options = None
        if k < 0:
            ccitt_group = 4  # Group 4
        elif k > 0:
            ccitt_group = 3  # Group 3 2-D
            t4_options = 1
        else:
            ccitt_group = 3  # Group 3 1-D
        black_is_one = self.decode_parms[0].get("/BlackIs1", False)
        decode = self._decode_array
        # PDF spec says:
        # BlackIs1: A flag indicating whether 1 bits shall be interpreted as black
        # pixels and 0 bits as white pixels, the reverse of the normal
        # PDF convention for image data. Default value: false.
        # TIFF spec says:
        # use 0 for white_is_zero (=> black is 1) MINISWHITE
        # use 1 for black_is_zero (=> white is 1) MINISBLACK
        photometry = 1 if black_is_one else 0

        # If Decode is [1, 0] then the photometry is inverted
        if len(decode) == 2 and decode == (1.0, 0.0):
            photometry = 1 - photometry

        img_size = len(data)
        if icc is None:
            icc = b''

        return _transcoding.generate_ccitt_header(
            self.size,
            data_length=img_size,
            ccitt_group=ccitt_group,
            t4_options=t4_options,
            photometry=photometry,
            icc=icc,
        )

    def show(self):  # pragma: no cover
        """Show the image however PIL wants to."""
        self.as_pil_image().show()

    def _set_pdf_source(self, pdf: Pdf):
        self._pdf_source = pdf

    def __repr__(self):
        try:
            mode = self.mode
        except NotImplementedError:
            mode = '?'
        return (
            f'<pikepdf.PdfImage image mode={mode} '
            f'size={self.width}x{self.height} at {hex(id(self))}>'
        )


class PdfJpxImage(PdfImage):
    """Support class for JPEG 2000 images. Implements the same API as :class:`PdfImage`.

    If you call PdfImage(object_that_is_actually_jpeg2000_image), pikepdf will return
    this class instead, due to the check in PdfImage.__new__.
    """

    def __init__(self, obj):
        """Initialize a JPEG 2000 image."""
        super().__init__(obj)
        self._jpxpil = self.as_pil_image()

    def __eq__(self, other):
        if not isinstance(other, PdfImageBase):
            return NotImplemented
        return (
            self.obj == other.obj
            and isinstance(other, PdfJpxImage)
            and self._jpxpil == other._jpxpil
        )

    def _extract_direct(self, *, stream: BinaryIO) -> str | None:
        data, filters = self._remove_simple_filters()
        if filters != ['/JPXDecode']:
            return None
        stream.write(data)
        return '.jp2'

    def _extract_transcoded(self) -> Image.Image:
        return super()._extract_transcoded()

    @property
    def _colorspaces(self):
        """Return the effective colorspace of a JPEG 2000 image.

        If the ColorSpace dictionary is present, the colorspace embedded in the
        JPEG 2000 data will be ignored, as required by the specification.
        """
        # (PDF 1.7 Table 89) If ColorSpace is present, any colour space
        # specifications in the JPEG2000 data shall be ignored.
        super_colorspaces = super()._colorspaces
        if super_colorspaces:
            return super_colorspaces
        if self._jpxpil.mode == 'L':
            return ['/DeviceGray']
        if self._jpxpil.mode == 'RGB':
            return ['/DeviceRGB']
        raise NotImplementedError('Complex JP2 colorspace')

    @property
    def _bpc(self) -> int:
        """Return 8, since bpc is not meaningful for JPEG 2000 encoding."""
        # (PDF 1.7 Table 89) If the image stream uses the JPXDecode filter, this
        # entry is optional and shall be ignored if present. The bit depth is
        # determined by the conforming reader in the process of decoding the
        # JPEG2000 image.
        return 8

    @property
    def indexed(self) -> bool:
        """Return False, since JPEG 2000 should not be indexed."""
        # Nothing in the spec precludes an Indexed JPXDecode image, except for
        # the fact that doing so is madness. Let's assume it no one is that
        # insane.
        return False

    def __repr__(self):
        return (
            f'<pikepdf.PdfJpxImage JPEG2000 image mode={self.mode} '
            f'size={self.width}x{self.height} at {hex(id(self))}>'
        )


class PdfInlineImage(PdfImageBase):
    """Support class for PDF inline images."""

    # Inline images can contain abbreviations that we write automatically
    ABBREVS = {
        b'/W': b'/Width',
        b'/H': b'/Height',
        b'/BPC': b'/BitsPerComponent',
        b'/IM': b'/ImageMask',
        b'/CS': b'/ColorSpace',
        b'/F': b'/Filter',
        b'/DP': b'/DecodeParms',
        b'/G': b'/DeviceGray',
        b'/RGB': b'/DeviceRGB',
        b'/CMYK': b'/DeviceCMYK',
        b'/I': b'/Indexed',
        b'/AHx': b'/ASCIIHexDecode',
        b'/A85': b'/ASCII85Decode',
        b'/LZW': b'/LZWDecode',
        b'/RL': b'/RunLengthDecode',
        b'/CCF': b'/CCITTFaxDecode',
        b'/DCT': b'/DCTDecode',
    }
    REVERSE_ABBREVS = {v: k for k, v in ABBREVS.items()}

    _data: Object
    _image_object: tuple[Object, ...]

    def __init__(self, *, image_data: Object, image_object: tuple):
        """Construct wrapper for inline image.

        Args:
            image_data: data stream for image, extracted from content stream
            image_object: the metadata for image, also from content stream
        """
        # Convert the sequence of pikepdf.Object from the content stream into
        # a dictionary object by unparsing it (to bytes), eliminating inline
        # image abbreviations, and constructing a bytes string equivalent to
        # what an image XObject would look like. Then retrieve data from there

        self._data = image_data
        self._image_object = image_object

        reparse = b' '.join(
            self._unparse_obj(obj, remap_names=self.ABBREVS) for obj in image_object
        )
        try:
            reparsed_obj = Object.parse(b'<< ' + reparse + b' >>')
        except PdfError as e:
            raise PdfError("parsing inline " + reparse.decode('unicode_escape')) from e
        self.obj = reparsed_obj

    def __eq__(self, other):
        if not isinstance(other, PdfImageBase):
            return NotImplemented
        return (
            self.obj == other.obj
            and isinstance(other, PdfInlineImage)
            and (
                self._data._inline_image_raw_bytes()
                == other._data._inline_image_raw_bytes()
            )
        )

    @classmethod
    def _unparse_obj(cls, obj, remap_names):
        if isinstance(obj, Object):
            if isinstance(obj, Name):
                name = obj.unparse(resolved=True)
                assert isinstance(name, bytes)
                return remap_names.get(name, name)
            return obj.unparse(resolved=True)
        if isinstance(obj, bool):
            return b'true' if obj else b'false'  # Lower case for PDF spec
        if isinstance(obj, (int, Decimal, float)):
            return str(obj).encode('ascii')
        raise NotImplementedError(repr(obj))

    def _metadata(self, name, type_, default):
        return _metadata_from_obj(self.obj, name, type_, default)

    def unparse(self) -> bytes:
        """Create the content stream bytes that reproduce this inline image."""

        def metadata_tokens():
            for metadata_obj in self._image_object:
                unparsed = self._unparse_obj(
                    metadata_obj, remap_names=self.REVERSE_ABBREVS
                )
                assert isinstance(unparsed, bytes)
                yield unparsed

        def inline_image_tokens():
            yield b'BI\n'
            yield b' '.join(m for m in metadata_tokens())
            yield b'\nID\n'
            yield self._data._inline_image_raw_bytes()
            yield b'EI'

        return b''.join(inline_image_tokens())

    @property
    def icc(self):  # pragma: no cover
        """Raise an exception since ICC profiles are not supported on inline images."""
        raise InvalidPdfImageError(
            "Inline images with ICC profiles are not supported in the PDF specification"
        )

    def __repr__(self):
        try:
            mode = self.mode
        except NotImplementedError:
            mode = '?'
        return (
            f'<pikepdf.PdfInlineImage image mode={mode} '
            f'size={self.width}x{self.height} at {hex(id(self))}>'
        )

    def _convert_to_pdfimage(self) -> PdfImage:
        # Construct a temporary PDF that holds this inline image, and...
        tmppdf = Pdf.new()
        tmppdf.add_blank_page(page_size=(self.width, self.height))
        tmppdf.pages[0].contents_add(
            f'{self.width} 0 0 {self.height} 0 0 cm'.encode('ascii'), prepend=True
        )
        tmppdf.pages[0].contents_add(self.unparse())

        # ...externalize it,
        tmppdf.pages[0].externalize_inline_images()
        raw_img = cast(Stream, next(im for im in tmppdf.pages[0].images.values()))

        # ...then use the regular PdfImage API to extract it.
        img = PdfImage(raw_img)
        img._set_pdf_source(tmppdf)  # Hold tmppdf open while PdfImage exists
        return img

    def as_pil_image(self) -> Image.Image:
        """Return inline image as a Pillow Image."""
        return self._convert_to_pdfimage().as_pil_image()

    def extract_to(self, *, stream: BinaryIO | None = None, fileprefix: str = ''):
        """Extract the inline image directly to a usable image file.

        See:
            :meth:`PdfImage.extract_to`
        """
        return self._convert_to_pdfimage().extract_to(
            stream=stream, fileprefix=fileprefix
        )

    def read_bytes(self):
        """Return decompressed image bytes."""
        # qpdf does not have an API to return this directly, so convert it.
        return self._convert_to_pdfimage().read_bytes()

    def get_stream_buffer(self):
        """Return decompressed stream buffer."""
        # qpdf does not have an API to return this directly, so convert it.
        return self._convert_to_pdfimage().get_stream_buffer()
