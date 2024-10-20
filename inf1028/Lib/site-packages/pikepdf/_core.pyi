# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

# pybind11 does not generate type annotations yet, and mypy doesn't understand
# the way we're augmenting C++ classes with Python methods as in
# pikepdf/_methods.py. Thus, we need to manually spell out the resulting types
# after augmenting.
import datetime
from abc import abstractmethod
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    ClassVar,
    Collection,
    Iterable,
    Iterator,
    KeysView,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    import numpy as np

    from pikepdf.models.encryption import Encryption, EncryptionInfo, Permissions
    from pikepdf.models.image import PdfInlineImage
    from pikepdf.models.metadata import PdfMetadata
    from pikepdf.models.outlines import Outline
    from pikepdf.objects import Array, Dictionary, Name, Operator, Stream, String

# This is the whole point of stub files, but apparently we have to do this...
# pylint: disable=no-method-argument,unused-argument,no-self-use,too-many-public-methods

# Rule: Function decorated with `@overload` shouldn't contain a docstring
# ruff: noqa: D418
# Seems to be no alternative for the moment.

# mypy: disable-error-code="misc"

T = TypeVar('T', bound='Object')
Numeric = TypeVar('Numeric', int, float, Decimal)

class Buffer:
    """A Buffer for reading data from a PDF."""

# Exceptions

class DataDecodingError(Exception):
    """Exception thrown when a stream object in a PDF cannot be decoded."""

class JobUsageError(Exception): ...

class PasswordError(Exception):
    """Exception thrown when the supplied password is incorrect."""

class PdfError(Exception):
    """General pikepdf-specific exception."""

class ForeignObjectError(Exception):
    """When a complex object is copied into a foreign PDF without proper methods.

    Use :meth:`Pdf.copy_foreign`.
    """

class DeletedObjectError(Exception):
    """When a required object is accessed after deletion.

    Thrown when accessing a :class:`Object` that relies on a :class:`Pdf`
    that was deleted using the Python ``delete`` statement or collected by the
    Python garbage collector. To resolve this error, you must retain a reference
    to the Pdf for the whole time you may be accessing it.

    .. versionadded:: 7.0
    """

# Enums
class AccessMode(Enum):
    default: int = ...
    mmap: int = ...
    mmap_only: int = ...
    stream: int = ...

class EncryptionMethod(Enum):
    """PDF encryption methods.

    Describes which encryption method was used on a particular part of a
    PDF. These values are returned by :class:`pikepdf.EncryptionInfo` but
    are not currently used to specify how encryption is requested.
    """

    none: int = ...
    """Data was not encrypted."""
    unknown: int = ...
    """An unknown algorithm was used."""
    rc4: int = ...
    """The RC4 encryption algorithm was used (obsolete)."""
    aes: int = ...
    """The AES-based algorithm was used as described in the |pdfrm|."""
    aesv3: int = ...
    """An improved version of the AES-based algorithm was used as described in the
        :doc:`Adobe Supplement to the ISO 32000 </references/resources>`, requiring
        PDF 1.7 extension level 3. This algorithm still uses AES, but allows both
        AES-128 and AES-256, and improves how the key is derived from the password."""

class ObjectStreamMode(Enum):
    """Options for saving object streams within PDFs.

    Object streams are more a compact
    way of saving certain types of data that was added in PDF 1.5. All
    modern PDF viewers support object streams, but some third party tools
    and libraries cannot read them.
    """

    disable: int = ...
    """Disable the use of object streams.

    If any object streams exist in the file, remove them when the file is saved.
    """
    generate: int = ...
    """Preserve any existing object streams in the original file.

    This is the default behavior.
    """
    preserve: int = ...
    """Generate object streams."""

class ObjectType(Enum):
    """Enumeration of PDF object types.

    These values are used to implement
    pikepdf's instance type checking. In the vast majority of cases it is more
    pythonic to use ``isinstance(obj, pikepdf.Stream)`` or ``issubclass``.

    These values are low-level and documented for completeness. They are exposed
    through :attr:`pikepdf.Object._type_code`.
    """

    array: int = ...
    """A PDF array, meaning the object is a ``pikepdf.Array``."""
    boolean: int = ...
    """A PDF boolean. In most cases, booleans are automatically converted to
        ``bool``, so this should not appear."""
    dictionary: int = ...
    """A PDF dictionary, meaning the object is a ``pikepdf.Dictionary``."""
    inlineimage: int = ...
    """A PDF inline image, meaning the object is the data stream of an inline
        image. It would be necessary to combine this with the implicit
        dictionary to interpret the image correctly. pikepdf automatically
        packages inline images into a more useful class, so this will not
        generally appear."""
    integer: int = ...
    """A PDF integer. In most cases, integers are automatically converted to
        ``int``, so this should not appear. Unlike Python integers, PDF integers
        are 32-bit signed integers."""
    name_: int = ...
    """A PDF name, meaning the object is a ``pikepdf.Name``."""
    null: int = ...
    """A PDF null. In most cases, nulls are automatically converted to ``None``,
        so this should not appear."""
    operator: int = ...
    """A PDF operator, meaning the object is a ``pikepdf.Operator``."""
    real: int = ...
    """A PDF real. In most cases, reals are automatically convert to
        :class:`decimal.Decimal`."""
    reserved: int = ...
    """A temporary object used in creating circular references. Should not appear
        in most cases."""
    stream: int = ...
    """A PDF stream, meaning the object is a ``pikepdf.Stream`` (and it also
        has a dictionary)."""
    string: int = ...
    """A PDF string, meaning the object is a ``pikepdf.String``."""
    uninitialized: int = ...
    """An uninitialized object. If this appears, it is probably a bug."""

class StreamDecodeLevel(Enum):
    """Options for decoding streams within PDFs."""

    all: int = ...
    """Do not attempt to apply any filters. Streams
        remain as they appear in the original file. Note that
        uncompressed streams may still be compressed on output. You can
        disable that by saving with ``.save(..., compress_streams=False)``."""
    generalized: int = ...
    """This is the default. libqpdf will apply
        LZWDecode, ASCII85Decode, ASCIIHexDecode, and FlateDecode
        filters on the input. When saved with
        ``compress_streams=True``, the default, the effect of this
        is that streams filtered with these older and less efficient
        filters will be recompressed with the Flate filter. As a
        special case, if a stream is already compressed with
        FlateDecode and ``compress_streams=True``, the original
        compressed data will be preserved."""
    none: int = ...
    """In addition to uncompressing the
        generalized compression formats, supported non-lossy
        compression will also be be decoded. At present, this includes
        the RunLengthDecode filter."""
    specialized: int = ...
    """        In addition to generalized and non-lossy
        specialized filters, supported lossy compression filters will
        be applied. At present, this includes DCTDecode (JPEG)
        compression. Note that compressing the resulting data with
        DCTDecode again will accumulate loss, so avoid multiple
        compression and decompression cycles. This is mostly useful for
        (low-level) retrieving image data; see :class:`pikepdf.PdfImage` for
        the preferred method."""

class TokenType(Enum):
    """Type of a token that appeared in a PDF content stream.

    When filtering content streams, each token is labeled according to the role
    in plays.
    """

    array_close: int = ...
    """The token data represents the end of an array."""
    array_open: int = ...
    """The token data represents the start of an array."""
    bad: int = ...
    """An invalid token."""
    bool: int = ...
    """The token data represents an integer, real number, null or boolean,
        respectively."""
    brace_close: int = ...
    """The token data represents the end of a brace."""
    brace_open: int = ...
    """The token data represents the start of a brace."""
    comment: int = ...
    """Signifies a comment that appears in the content stream."""
    dict_close: int = ...
    """The token data represents the end of a dictionary."""
    dict_open: int = ...
    """The token data represents the start of a dictionary."""
    eof: int = ...
    """Denotes the end of the tokens in this content stream."""
    inline_image: int = ...
    """An inline image in the content stream. The whole inline image is
        represented by the single token."""
    integer: int = ...
    """The token data represents an integer."""
    name_: int = ...
    """The token is the name (pikepdf.Name) of an object. In practice, these
        are among the most interesting tokens.

        .. versionchanged:: 3.0
            In versions older than 3.0, ``.name`` was used instead. This interfered
            with semantics of the ``Enum`` object, so this was fixed.
    """
    null: int = ...
    """The token data represents a null."""
    real: int = ...
    """The token data represents a real number."""
    space: int = ...
    """Whitespace within the content stream."""
    string: int = ...
    """The token data represents a string. The encoding is unclear and situational."""
    word: int = ...
    """Otherwise uncategorized bytes are returned as ``word`` tokens. PDF
        operators are words."""

class Object:
    def _ipython_key_completions_(self) -> KeysView | None: ...
    def _inline_image_raw_bytes(self) -> bytes: ...
    def _parse_page_contents(self, callbacks: Callable) -> None: ...
    def _parse_page_contents_grouped(
        self, whitelist: str
    ) -> list[tuple[Collection[Object | PdfInlineImage], Operator]]: ...
    @staticmethod
    def _parse_stream(stream: Object, parser: StreamParser) -> list: ...
    @staticmethod
    def _parse_stream_grouped(stream: Object, whitelist: str) -> list: ...
    def _repr_mimebundle_(self, include=None, exclude=None) -> dict | None: ...
    def _write(
        self,
        data: bytes,
        filter: Object,  # pylint: disable=redefined-builtin
        decode_parms: Object,
    ) -> None: ...
    def append(self, pyitem: Any) -> None:
        """Append another object to an array; fails if the object is not an array."""
    def as_dict(self) -> _ObjectMapping: ...
    def as_list(self) -> _ObjectList: ...
    def emplace(self, other: Object, retain: Iterable[Name] = ...) -> None:
        """Copy all items from other without making a new object.

        Particularly when working with pages, it may be desirable to remove all
        of the existing page's contents and emplace (insert) a new page on top
        of it, in a way that preserves all links and references to the original
        page. (Or similarly, for other Dictionary objects in a PDF.)

        Any Dictionary keys in the iterable *retain* are preserved. By default,
        /Parent is retained.

        When a page is assigned (``pdf.pages[0] = new_page``), only the
        application knows if references to the original the original page are
        still valid. For example, a PDF optimizer might restructure a page
        object into another visually similar one, and references would be valid;
        but for a program that reorganizes page contents such as a N-up
        compositor, references may not be valid anymore.

        This method takes precautions to ensure that child objects in common
        with ``self`` and ``other`` are not inadvertently deleted.

        Example:
            >>> pdf = pikepdf.Pdf.open('../tests/resources/fourpages.pdf')
            >>> pdf.pages[0].objgen
            (3, 0)
            >>> pdf.pages[0].emplace(pdf.pages[1])
            >>> pdf.pages[0].objgen
            (3, 0)
            >>> # Same object

        .. versionchanged:: 2.11.1
            Added the *retain* argument.
        """
    def extend(self, iter: Iterable[Object]) -> None:
        """Extend a pikepdf.Array with an iterable of other pikepdf.Object."""
    def get(self, key: int | str | Name, default: T | None = ...) -> Object | T | None:
        """Retrieve an attribute from the object.

        Only works if the object is a Dictionary, Array or Stream.
        """
    def get_raw_stream_buffer(self) -> Buffer:
        """Return a buffer protocol buffer describing the raw, encoded stream."""
    def get_stream_buffer(self, decode_level: StreamDecodeLevel = ...) -> Buffer:
        """Return a buffer protocol buffer describing the decoded stream."""
    def is_owned_by(self, possible_owner: Pdf) -> bool:
        """Test if this object is owned by the indicated *possible_owner*."""
    def items(self) -> Iterable[tuple[str, Object]]: ...
    def keys(self) -> set[str]:
        """Get the keys of the object, if it is a Dictionary or Stream."""
    @staticmethod
    def parse(stream: bytes, description: str = ...) -> Object:
        """Parse PDF binary representation into PDF objects."""
    def read_bytes(self, decode_level: StreamDecodeLevel = ...) -> bytes:
        """Decode and read the content stream associated with this object."""
    def read_raw_bytes(self) -> bytes:
        """Read the content stream associated with a Stream, without decoding."""
    def same_owner_as(self, other: Object) -> bool:
        """Test if two objects are owned by the same :class:`pikepdf.Pdf`."""
    def to_json(self, dereference: bool = ..., schema_version: int = ...) -> bytes:
        r"""Convert to a qpdf JSON representation of the object.

        See the qpdf manual for a description of its JSON representation.
        https://qpdf.readthedocs.io/en/stable/json.html#qpdf-json-format

        Not necessarily compatible with other PDF-JSON representations that
        exist in the wild.

        * Names are encoded as UTF-8 strings
        * Indirect references are encoded as strings containing ``obj gen R``
        * Strings are encoded as UTF-8 strings with unrepresentable binary
            characters encoded as ``\uHHHH``
        * Encoding streams just encodes the stream's dictionary; the stream
            data is not represented
        * Object types that are only valid in content streams (inline
            image, operator) as well as "reserved" objects are not
            representable and will be serialized as ``null``.

        Args:
            dereference (bool): If True, dereference the object if this is an
                indirect object.
            schema_version (int): The version of the JSON schema. Defaults to 2.

        Returns:
            JSON bytestring of object. The object is UTF-8 encoded
            and may be decoded to a Python str that represents the binary
            values ``\x00-\xFF`` as ``U+0000`` to ``U+00FF``; that is,
            it may contain mojibake.

        .. versionchanged:: 6.0
            Added *schema_version*.
        """
    def unparse(self, resolved: bool = ...) -> bytes:
        """Convert PDF objects into their binary representation.

        Set resolved=True to deference indirect objects where possible.

        If you want to unparse content streams, which are a collection of
        objects that need special treatment, use
        :func:`pikepdf.unparse_content_stream` instead.

        Returns ``bytes()`` that can be used with :meth:`Object.parse`
        to reconstruct the ``pikepdf.Object``. If reconstruction is not possible,
        a relative object reference is returned, such as ``4 0 R``.

        Args:
            resolved: If True, deference indirect objects where possible.
        """
    def with_same_owner_as(self, arg0: Object) -> Object:
        """Returns an object that is owned by the same Pdf that owns *other* object.

        If the objects already have the same owner, this object is returned.
        If the *other* object has a different owner, then a copy is created
        that is owned by *other*'s owner. If this object is a direct object
        (no owner), then an indirect object is created that is owned by
        *other*. An exception is thrown if *other* is a direct object.

        This method may be convenient when a reference to the Pdf is not
        available.

        .. versionadded:: 2.14
        """
    def wrap_in_array(self) -> Array:
        """Return the object wrapped in an array if not already an array."""
    def write(
        self,
        data: bytes,
        *,
        filter: Name | Array | list[Name] | None = ...,  # pylint: disable=redefined-builtin
        decode_parms: Dictionary | Array | None = ...,
        type_check: bool = ...,
    ) -> None:
        """Replace stream object's data with new (possibly compressed) `data`.

        `filter` and `decode_parms` describe any compression that is already
        present on the input `data`. For example, if your data is already
        compressed with the Deflate algorithm, you would set
        ``filter=Name.FlateDecode``.

        When writing the PDF in :meth:`pikepdf.Pdf.save`,
        pikepdf may change the compression or apply compression to data that was
        not compressed, depending on the parameters given to that function. It
        will never change lossless to lossy encoding.

        PNG and TIFF images, even if compressed, cannot be directly inserted
        into a PDF and displayed as images.

        Args:
            data: the new data to use for replacement
            filter: The filter(s) with which the
                data is (already) encoded
            decode_parms: Parameters for the
                filters with which the object is encode
            type_check: Check arguments; use False only if you want to
                intentionally create malformed PDFs.

        If only one `filter` is specified, it may be a name such as
        `Name('/FlateDecode')`. If there are multiple filters, then array
        of names should be given.

        If there is only one filter, `decode_parms` is a Dictionary of
        parameters for that filter. If there are multiple filters, then
        `decode_parms` is an Array of Dictionary, where each array index
        is corresponds to the filter.
        """
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __contains__(self, obj: Object | str) -> bool: ...
    def __copy__(self) -> Object: ...
    def __delattr__(self, name: str) -> None: ...
    def __delitem__(self, name: str | Name | int) -> None: ...
    def __dir__(self) -> list: ...
    def __eq__(self, other: Any) -> bool: ...
    def __float__(self) -> float: ...
    def __getattr__(self, name: str) -> Object: ...
    def __getitem__(self, name: str | Name | int) -> Object: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __iter__(self) -> Iterable[Object]: ...
    def __len__(self) -> int: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __setitem__(self, name: str | Name | int, value: Any) -> None: ...
    @property
    def _objgen(self) -> tuple[int, int]: ...
    @property
    def _type_code(self) -> ObjectType: ...
    @property
    def _type_name(self) -> str: ...
    @property
    def images(self) -> _ObjectMapping: ...
    @property
    def is_indirect(self) -> bool:
        """Returns True if the object is an indirect object."""
    @property
    def is_rectangle(self) -> bool:
        """Returns True if the object is a rectangle (an array of 4 numbers)."""
    @property
    def objgen(self) -> tuple[int, int]:
        """Return the object-generation number pair for this object.

        If this is a direct object, then the returned value is ``(0, 0)``.
        By definition, if this is an indirect object, it has a "objgen",
        and can be looked up using this in the cross-reference (xref) table.
        Direct objects cannot necessarily be looked up.

        The generation number is usually 0, except for PDFs that have been
        incrementally updated. Incrementally updated PDFs are now uncommon,
        since it does not take too long for modern CPUs to reconstruct an
        entire PDF. pikepdf will consolidate all incremental updates
        when saving.
        """
    @property
    def stream_dict(self) -> Dictionary:
        """Access the dictionary key-values for a :class:`pikepdf.Stream`."""
    @stream_dict.setter
    def stream_dict(self, val: Dictionary) -> None: ...

class ObjectHelper:
    """Base class for wrapper/helper around an Object.

    Used to expose additional functionality specific to that object type.

    :class:`pikepdf.Page` is an example of an object helper. The actual
    page object is a PDF is a Dictionary. The helper provides additional
    methods specific to pages.
    """

    def __eq__(self, other: Any) -> bool: ...
    @property
    def obj(self) -> Dictionary:
        """Get the underlying PDF object (typically a Dictionary)."""

class _ObjectList:
    """A list whose elements are always pikepdf.Object.

    In all other respects, this object behaves like a standard Python
    list.
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: _ObjectList) -> None: ...
    @overload
    def __init__(self, arg0: Iterable) -> None: ...
    @overload
    def __init__(*args, **kwargs) -> None: ...
    def append(self, x: Object) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: Object) -> int: ...
    @overload
    def extend(self, L: _ObjectList) -> None: ...
    @overload
    def extend(self, L: Iterable[Object]) -> None: ...
    def insert(self, i: int, x: Object) -> None: ...
    @overload
    def pop(self) -> Object: ...
    @overload
    def pop(self, i: int) -> Object: ...
    @overload
    def pop(*args, **kwargs) -> Any: ...
    def remove(self, x: Object) -> None: ...
    def __bool__(self) -> bool: ...
    def __contains__(self, x: Object) -> bool: ...
    @overload
    def __delitem__(self, arg0: int) -> None: ...
    @overload
    def __delitem__(self, arg0: slice) -> None: ...
    @overload
    def __delitem__(*args, **kwargs) -> Any: ...
    def __eq__(self, other: Any) -> bool: ...
    @overload
    def __getitem__(self, s: slice) -> _ObjectList: ...
    @overload
    def __getitem__(self, arg0: int) -> Object: ...
    @overload
    def __getitem__(*args, **kwargs) -> Any: ...
    def __iter__(self) -> Iterator[Object]: ...
    def __len__(self) -> int: ...
    def __ne__(self, other: Any) -> bool: ...
    @overload
    def __setitem__(self, arg0: int, arg1: Object) -> None: ...
    @overload
    def __setitem__(self, arg0: slice, arg1: _ObjectList) -> None: ...
    @overload
    def __setitem__(*args, **kwargs) -> Any: ...

class _ObjectMapping:
    """A mapping whose keys and values are always pikepdf.Name and pikepdf.Object."""

    def get(self, key: Name | str, default: T = ...) -> Object | T: ...
    def keys(self) -> Iterator[Name]: ...
    def values(self) -> Iterator[Object]: ...
    def __contains__(self, key: Name | str) -> bool: ...
    def __init__(self) -> None: ...
    def items(self) -> Iterator: ...
    def __bool__(self) -> bool: ...
    def __delitem__(self, key: str) -> None: ...
    def __getitem__(self, key: Name | str) -> Object: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: str, value: Object) -> None: ...

class Annotation:
    """A PDF annotation. Wrapper around a PDF dictionary.

    Describes an annotation in a PDF, such as a comment, underline,
    copy editing marks, interactive widgets, redactions, 3D objects, sound
    and video clips.

    See the |pdfrm| section 12.5.6 for the full list of annotation types
    and definition of terminology.

    .. versionadded:: 2.12
    """

    def __init__(self, obj: Object) -> None: ...
    def get_appearance_stream(
        self, which: Object, state: Object | None = ...
    ) -> Object:
        """Returns one of the appearance streams associated with an annotation.

        Args:
            which: Usually one of ``pikepdf.Name.N``, ``pikepdf.Name.R`` or
                ``pikepdf.Name.D``, indicating the normal, rollover or down
                appearance stream, respectively. If any other name is passed,
                an appearance stream with that name is returned.
            state: The appearance state. For checkboxes or radio buttons, the
                appearance state is usually whether the button is on or off.
        """
    def get_page_content_for_appearance(
        self,
        name: Name,
        rotate: int,
        required_flags: int = ...,
        forbidden_flags: int = ...,
    ) -> bytes:
        """Generate content stream text that draws this annotation as a Form XObject.

        Args:
            name: What to call the object we create.
            rotate: Should be set to the page's /Rotate value or 0.
            required_flags: The required appearance flags. See PDF reference manual.
            forbidden_flags: The forbidden appearance flags. See PDF reference manual.

        Note:
            This method is done mainly with qpdf. Its behavior may change when
            different qpdf versions are used.
        """
    @property
    def appearance_dict(self) -> Object:
        """Returns the annotations appearance dictionary."""
    @property
    def appearance_state(self) -> Object:
        """Returns the annotation's appearance state (or None).

        For a checkbox or radio button, the appearance state may be ``pikepdf.Name.On``
        or ``pikepdf.Name.Off``.
        """
    @property
    def flags(self) -> int:
        """Returns the annotation's flags."""
    @property
    def obj(self) -> Object: ...
    @property
    def subtype(self) -> str:
        """Returns the subtype of this annotation."""

class AttachedFile:
    """An object that contains an actual attached file.

    These objects do not need to be created manually; they are normally part of an
    AttachedFileSpec.

    .. versionadded:: 3.0
    """

    _creation_date: str
    _mod_date: str
    creation_date: datetime.datetime | None
    mime_type: str
    """Get the MIME type of the attached file according to the PDF creator."""
    mod_date: datetime.datetime | None
    @property
    def md5(self) -> bytes:
        """Get the MD5 checksum of attached file according to the PDF creator."""
    @property
    def obj(self) -> Object: ...
    def read_bytes(self) -> bytes: ...
    @property
    def size(self) -> int:
        """Get length of the attached file in bytes according to the PDF creator."""

class AttachedFileSpec(ObjectHelper):
    r"""In a PDF, a file specification provides name and metadata for a target file.

    Most file specifications are *simple* file specifications, and contain only
    one attached file. Call :meth:`get_file` to get the attached file:

    .. code-block:: python

        pdf = Pdf.open(...)

        fs = pdf.attachments['example.txt']
        stream = fs.get_file()

    To attach a new file to a PDF, you may construct a ``AttachedFileSpec``.

    .. code-block:: python

        pdf = Pdf.open(...)

        fs = AttachedFileSpec.from_filepath(pdf, Path('somewhere/spreadsheet.xlsx'))

        pdf.attachments['spreadsheet.xlsx'] = fs

    PDF supports the concept of having multiple, platform-specialized versions of the
    attached file (similar to resource forks on some operating systems). In theory,
    this attachment ought to be the same file, but
    encoded in different ways. For example, perhaps a PDF includes a text file encoded
    with Windows line endings (``\r\n``) and a different one with POSIX line endings
    (``\n``). Similarly, PDF allows for the possibility that you need to encode
    platform-specific filenames. pikepdf cannot directly create these, because they
    are arguably obsolete; it can provide access to them, however.

    If you have to deal with platform-specialized versions,
    use :meth:`get_all_filenames` to enumerate those available.

    Described in the |pdfrm| section 7.11.3.

    .. versionadded:: 3.0
    """

    def __init__(
        self,
        data: bytes,
        *,
        description: str,
        filename: str,
        mime_type: str,
        creation_date: str,
        mod_date: str,
    ) -> None:
        """Construct a attached file spec from data in memory.

        To construct a file spec from a file on the computer's file system,
        use :meth:`from_filepath`.

        Args:
            data: Resource to load.
            description: Any description text for the attachment. May be
                shown in PDF viewers.
            filename: Filename to display in PDF viewers.
            mime_type: Helps PDF viewers decide how to display the information.
            creation_date: PDF date string for when this file was created.
            mod_date: PDF date string for when this file was last modified.
            relationship: A :class:`pikepdf.Name` indicating the relationship
                of this file to the document. Canonically, this should be a name
                from the PDF specification:
                Source, Data, Alternative, Supplement, EncryptedPayload, FormData,
                Schema, Unspecified. If omitted, Unspecified is used.
        """
    def get_all_filenames(self) -> dict:
        """Return a Python dictionary that describes all filenames.

        The returned dictionary is not a pikepdf Object.

        Multiple filenames are generally a holdover from the pre-Unicode era.
        Modern PDFs can generally set UTF-8 filenames and avoid using
        punctuation or other marks that are forbidden in filenames.
        """
    def get_file(self, name: Name = ...) -> AttachedFile:
        """Return an attached file.

        Typically, only one file is attached to an attached file spec.
        When multiple files are attached, use the ``name`` parameter to
        specify which one to return.

        Args:
            name: Typical names would be ``/UF`` and ``/F``. See |pdfrm|
                for other obsolete names.
        """
    @staticmethod
    def from_filepath(
        pdf: Pdf, path: Path | str, *, description: str = ''
    ) -> AttachedFileSpec:
        """Construct a file specification from a file path.

        This function will automatically add a creation and modified date
        using the file system, and a MIME type inferred from the file's extension.

        If the data required for the attach is in memory, use
        :meth:`pikepdf.AttachedFileSpec` instead.

        Args:
            pdf: The Pdf to attach this file specification to.
            path: A file path for the file to attach to this Pdf.
            description: An optional description. May be shown to the user in
                PDF viewers.
            relationship: An optional relationship type. May be used to
                indicate the type of attachment, e.g. Name.Source or Name.Data.
                Canonically, this should be a name from the PDF specification:
                Source, Data, Alternative, Supplement, EncryptedPayload, FormData,
                Schema, Unspecified. If omitted, Unspecified is used.
        """
    @property
    def description(self) -> str:
        """Description text associated with the embedded file."""
    @property
    def filename(self) -> str:
        """The main filename for this file spec.

        In priority order, getting this returns the first of /UF, /F, /Unix,
        /DOS, /Mac if multiple filenames are set. Setting this will set a UTF-8
        encoded Unicode filename and write it to /UF.
        """
    @property
    def relationship(self) -> Name | None:
        """Describes the relationship of this attached file to the PDF."""
    @relationship.setter
    def relationship(self, value: Name | None) -> None: ...

class Attachments(MutableMapping[str, AttachedFileSpec]):
    """Exposes files attached to a PDF.

    If a file is attached to a PDF, it is exposed through this interface.
    For example ``p.attachments['readme.txt']`` would return a
    :class:`pikepdf._core.AttachedFileSpec` that describes the attached file,
    if a file were attached under that name.
    ``p.attachments['readme.txt'].get_file()`` would return a
    :class:`pikepdf._core.AttachedFile`, an archaic intermediate object to support
    different versions of the file for different platforms. Typically one
    just calls ``p.attachments['readme.txt'].read_bytes()`` to get the
    contents of the file.

    This interface provides access to any files that are attached to this PDF,
    exposed as a Python :class:`collections.abc.MutableMapping` interface.

    The keys (virtual filenames) are always ``str``, and values are always
    :class:`pikepdf.AttachedFileSpec`.

    To create a new attached file, use
    :meth:`pikepdf._core.AttachedFileSpec.from_filepath`
    to create a :class:`pikepdf._core.AttachedFileSpec` and then assign it to the
    :attr:`pikepdf.Pdf.attachments` mapping. If the file is in memory, use
    ``p.attachments['test.pdf'] = b'binary data'``.

    Use this interface through :attr:`pikepdf.Pdf.attachments`.

    .. versionadded:: 3.0

    .. versionchanged:: 8.10.1
        Added convenience interface for directly loading attached files, e.g.
        ``pdf.attachments['/test.pdf'] = b'binary data'``. Prior to this release,
        there was no way to attach data in memory as a file.
    """

    def __contains__(self, k: object) -> bool: ...
    def __delitem__(self, k: str) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __getitem__(self, k: str) -> AttachedFileSpec: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __setitem__(self, k: str, v: AttachedFileSpec | bytes): ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _add_replace_filespec(self, arg0: str, arg1: AttachedFileSpec) -> None: ...
    def _get_all_filespecs(self) -> dict[str, AttachedFileSpec]: ...
    def _get_filespec(self, arg0: str) -> AttachedFileSpec: ...
    def _remove_filespec(self, arg0: str) -> bool: ...
    @property
    def _has_embedded_files(self) -> bool: ...

class Token:
    def __init__(self, arg0: TokenType, arg1: bytes) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    @property
    def error_msg(self) -> str:
        """If the token is an error, this returns the error message."""
    @property
    def raw_value(self) -> bytes:
        """The binary representation of a token."""
    @property
    def type_(self) -> TokenType:
        """Returns the type of token."""
    @property
    def value(self) -> str:
        """Interprets the token as a string."""

class _QPDFTokenFilter: ...

class TokenFilter(_QPDFTokenFilter):
    def __init__(self) -> None: ...
    def handle_token(self, token: Token = ...) -> None | Token | Iterable[Token]:
        """Handle a :class:`pikepdf.Token`.

        This is an abstract method that must be defined in a subclass
        of ``TokenFilter``. The method will be called for each token.
        The implementation may return either ``None`` to discard the
        token, the original token to include it, a new token, or an
        iterable containing zero or more tokens. An implementation may
        also buffer tokens and release them in groups (for example, it
        could collect an entire PDF command with all of its operands,
        and then return all of it).

        The final token will always be a token of type ``TokenType.eof``,
        (unless an exception is raised).

        If this method raises an exception, the exception will be
        caught by C++, consumed, and replaced with a less informative
        exception. Use :meth:`pikepdf.Pdf.get_warnings` to view the
        original.
        """

class StreamParser:
    """A simple content stream parser, which must be subclassed to be used.

    In practice, the performance of this class may be quite poor on long
    content streams because it creates objects and involves multiple
    function calls for every object in a content stream, some of which
    may be only a single byte long.

    Consider instead using :func:`pikepdf.parse_content_stream`.
    """

    def __init__(self) -> None: ...
    @abstractmethod
    def handle_eof(self) -> None:
        """An abstract method that may be overloaded in a subclass.

        Called at the end of a content stream.
        """
    @abstractmethod
    def handle_object(self, obj: Object, offset: int, length: int) -> None:
        """An abstract method that must be overloaded in a subclass.

        This function will be called back once for each object that is
        parsed in the content stream.
        """

class Page:
    """Support model wrapper around a page dictionary object."""

    def _repr_mimebundle_(include: Any = ..., exclude: Any = ...) -> Any:
        """Present options to IPython or Jupyter for rich display of this object.

        See:
        https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display
        """
    @overload
    def __init__(self, arg0: Object) -> None: ...
    @overload
    def __init__(self, arg0: Page) -> None: ...
    def __contains__(self, key: Any) -> bool: ...
    def __delattr__(self, name: Any) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __getattr__(self, name: Any) -> Object: ...
    def __getitem__(self, name: Any) -> Object: ...
    def __setattr__(self, name: Any, value: Any): ...
    def __setitem__(self, name: Any, value: Any): ...
    def _get_artbox(self, arg0: bool, arg1: bool) -> Object: ...
    def _get_bleedbox(self, arg0: bool, arg1: bool) -> Object: ...
    def _get_cropbox(self, arg0: bool, arg1: bool) -> Object: ...
    def _get_mediabox(self, arg0: bool) -> Object: ...
    def _get_trimbox(self, arg0: bool, arg1: bool) -> Object: ...
    def add_content_token_filter(self, tf: TokenFilter) -> None:
        """Attach a :class:`pikepdf.TokenFilter` to a page's content stream.

        This function applies token filters lazily, if/when the page's
        content stream is read for any reason, such as when the PDF is
        saved. If never access, the token filter is not applied.

        Multiple token filters may be added to a page/content stream.

        Token filters may not be removed after being attached to a Pdf.
        Close and reopen the Pdf to remove token filters.

        If the page's contents is an array of streams, it is coalesced.

        Args:
            tf: The token filter to attach.
        """
    def add_overlay(
        self,
        other: Object | Page,
        rect: Rectangle | None,
        *,
        push_stack: bool | None = ...,
    ):
        """Overlay another object on this page.

        Overlays will be drawn after all previous content, potentially drawing on top
        of existing content.

        Args:
            other: A Page or Form XObject to render as an overlay on top of this
                page.
            rect: The PDF rectangle (in PDF units) in which to draw the overlay.
                If omitted, this page's trimbox, cropbox or mediabox (in that order)
                will be used.
            push_stack: If True (default), push the graphics stack of the existing
                content stream to ensure that the overlay is rendered correctly.
                Officially PDF limits the graphics stack depth to 32. Most
                viewers will tolerate more, but excessive pushes may cause problems.
                Multiple content streams may also be coalesced into a single content
                stream where this parameter is True, since the PDF specification
                permits PDF writers to coalesce streams as they see fit.
            shrink: If True (default), allow the object to shrink to fit inside the
                rectangle. The aspect ratio will be preserved.
            expand: If True (default), allow the object to expand to fit inside the
                rectangle. The aspect ratio will be preserved.

        Returns:
            The name of the Form XObject that contains the overlay.

        .. versionadded:: 2.14

        .. versionchanged:: 4.0.0
            Added the *push_stack* parameter. Previously, this method behaved
            as if *push_stack* were False.

        .. versionchanged:: 4.2.0
            Added the *shrink* and *expand* parameters. Previously, this method
            behaved as if ``shrink=True, expand=False``.

        .. versionchanged:: 4.3.0
            Returns the name of the overlay in the resources dictionary instead
            of returning None.
        """
    def add_underlay(self, other: Object | Page, rect: Rectangle | None):
        """Underlay another object beneath this page.

        Underlays will be drawn before all other content, so they may be overdrawn
        partially or completely.

        There is no *push_stack* parameter for this function, since adding an
        underlay can be done without manipulating the graphics stack.

        Args:
            other: A Page or Form XObject to render as an underlay underneath this
                page.
            rect: The PDF rectangle (in PDF units) in which to draw the underlay.
                If omitted, this page's trimbox, cropbox or mediabox (in that order)
                will be used.
            shrink: If True (default), allow the object to shrink to fit inside the
                rectangle. The aspect ratio will be preserved.
            expand: If True (default), allow the object to expand to fit inside the
                rectangle. The aspect ratio will be preserved.

        Returns:
            The name of the Form XObject that contains the underlay.

        .. versionadded:: 2.14

        .. versionchanged:: 4.2.0
            Added the *shrink* and *expand* parameters. Previously, this method
            behaved as if ``shrink=True, expand=False``. Fixed issue with wrong
            page rect being selected.
        """
    def as_form_xobject(self, handle_transformations: bool = ...) -> Object:
        """Return a form XObject that draws this page.

        This is useful for
        n-up operations, underlay, overlay, thumbnail generation, or
        any other case in which it is useful to replicate the contents
        of a page in some other context. The dictionaries are shallow
        copies of the original page dictionary, and the contents are
        coalesced from the page's contents. The resulting object handle
        is not referenced anywhere.

        Args:
            handle_transformations: If True (default), the resulting form
                XObject's ``/Matrix`` will be set to replicate rotation
                (``/Rotate``) and scaling (``/UserUnit``) in the page's
                dictionary. In this way, the page's transformations will
                be preserved when placing this object on another page.
        """
    def calc_form_xobject_placement(
        self,
        formx: Object,
        name: Name,
        rect: Rectangle,
        *,
        invert_transformations: bool,
        allow_shrink: bool,
        allow_expand: bool,
    ) -> bytes:
        """Generate content stream segment to place a Form XObject on this page.

        The content stream segment must then be added to the page's
        content stream.

        The default keyword parameters will preserve the aspect ratio.

        Args:
            formx: The Form XObject to place.
            name: The name of the Form XObject in this page's /Resources
                dictionary.
            rect: Rectangle describing the desired placement of the Form
                XObject.
            invert_transformations: Apply /Rotate and /UserUnit scaling
                when determining FormX Object placement.
            allow_shrink: Allow the Form XObject to take less than the
                full dimensions of rect.
            allow_expand: Expand the Form XObject to occupy all of rect.

        .. versionadded:: 2.14
        """
    def contents_add(self, contents: Stream | bytes, *, prepend: bool = ...) -> None:
        """Append or prepend to an existing page's content stream.

        Args:
            contents: An existing content stream to append or prepend.
            prepend: Prepend if true, append if false (default).

        .. versionadded:: 2.14
        """
    def contents_coalesce(self) -> None:
        """Coalesce a page's content streams.

        A page's content may be a
        stream or an array of streams. If this page's content is an
        array, concatenate the streams into a single stream. This can
        be useful when working with files that split content streams in
        arbitrary spots, such as in the middle of a token, as that can
        confuse some software.
        """
    def emplace(self, other: Page, retain: Iterable[Name] = ...) -> None: ...
    def externalize_inline_images(
        self, min_size: int = ..., shallow: bool = ...
    ) -> None:
        """Convert inline image to normal (external) images.

        Args:
            min_size: minimum size in bytes
            shallow: If False, recurse into nested Form XObjects.
                If True, do not recurse.
        """
    def form_xobjects(self) -> _ObjectMapping:
        """Return all Form XObjects associated with this page.

        This method does not recurse into nested Form XObjects.

        .. versionadded:: 7.0.0
        """
    def get(self, key: str | Name, default: T | None = ...) -> T | None | Object: ...
    def get_filtered_contents(self, tf: TokenFilter) -> bytes:
        """Apply a :class:`pikepdf.TokenFilter` to a content stream.

        This may be used when the results of a token filter do not need
        to be applied, such as when filtering is being used to retrieve
        information rather than edit the content stream.

        Note that it is possible to create a subclassed ``TokenFilter``
        that saves information of interest to its object attributes; it
        is not necessary to return data in the content stream.

        To modify the content stream, use :meth:`pikepdf.Page.add_content_token_filter`.

        Returns:
            The result of modifying the content stream with ``tf``.
            The existing content stream is not modified.
        """
    def index(self) -> int:
        """Returns the zero-based index of this page in the pages list.

        That is, returns ``n`` such that ``pdf.pages[n] == this_page``.
        A ``ValueError`` exception is thrown if the page is not attached
        to this ``Pdf``.

        .. versionadded:: 2.2
        """
    def label(self) -> str:
        """Returns the page label for this page, accounting for section numbers.

        For example, if the PDF defines a preface with lower case Roman
        numerals (i, ii, iii...), followed by standard numbers, followed
        by an appendix (A-1, A-2, ...), this function returns the appropriate
        label as a string.

        It is possible for a PDF to define page labels such that multiple
        pages have the same labels. Labels are not guaranteed to
        be unique.

        .. versionadded:: 2.2

        .. versionchanged:: 2.9
            Returns the ordinary page number if no special rules for page
            numbers are defined.
        """
    def parse_contents(self, stream_parser: StreamParser) -> None:
        """Parse a page's content streams using a :class:`pikepdf.StreamParser`.

        The content stream may be interpreted by the StreamParser but is
        not altered.

        If the page's contents is an array of streams, it is coalesced.

        Args:
            stream_parser: A :class:`pikepdf.StreamParser` instance.
        """
    def remove_unreferenced_resources(self) -> None:
        """Removes resources not referenced by content stream.

        A page's resources (``page.resources``) dictionary maps names to objects.
        This method walks through a page's contents and
        keeps tracks of which resources are referenced somewhere in the
        contents. Then it removes from the resources dictionary any
        object that is not referenced in the contents. This
        method is used by page splitting code to avoid copying unused
        objects in files that use shared resource dictionaries across
        multiple pages.
        """
    def rotate(self, angle: int, relative: bool) -> None:
        """Rotate a page.

        If ``relative`` is ``False``, set the rotation of the
        page to angle. Otherwise, add angle to the rotation of the
        page. ``angle`` must be a multiple of ``90``. Adding ``90`` to
        the rotation rotates clockwise by ``90`` degrees.

        Args:
            angle: Rotation angle in degrees.
            relative: If ``True``, add ``angle`` to the current
                rotation. If ``False``, set the rotation of the page
                to ``angle``.
        """
    @property
    def images(self) -> _ObjectMapping:
        """Return all regular images associated with this page.

        This method does not search for Form XObjects that contain images,
        and does not attempt to find inline images.
        """
    @property
    def artbox(self) -> Array:
        """Return page's effective /ArtBox, in PDF units.

        According to the PDF specification:
        "The art box defines the page's meaningful content area, including
        white space."

        If the /ArtBox is not defined, the /CropBox is returned.
        """
    @artbox.setter
    def artbox(self, val: Array | Rectangle) -> None: ...
    @property
    def bleedbox(self) -> Array:
        """Return page's effective /BleedBox, in PDF units.

        According to the PDF specification:
        "The bleed box defines the region to which the contents of the page
        should be clipped when output in a print production environment."

        If the /BleedBox is not defined, the /CropBox is returned.
        """
    @bleedbox.setter
    def bleedbox(self, val: Array | Rectangle) -> None: ...
    @property
    def cropbox(self) -> Array:
        """Return page's effective /CropBox, in PDF units.

        According to the PDF specification:
        "The crop box defines the region to which the contents of the page
        shall be clipped (cropped) when displayed or printed. It has no
        defined meaning in the context of the PDF imaging model; it merely
        imposes clipping on the page contents."

        If the /CropBox is not defined, the /MediaBox is returned.
        """
    @cropbox.setter
    def cropbox(self, val: Array | Rectangle) -> None: ...
    @property
    def mediabox(self) -> Array:
        """Return page's /MediaBox, in PDF units.

        According to the PDF specification:
        "The media box defines the boundaries of the physical medium on which
        the page is to be printed."
        """
    @mediabox.setter
    def mediabox(self, val: Array | Rectangle) -> None: ...
    @property
    def obj(self) -> Dictionary: ...
    @property
    def trimbox(self) -> Array:
        """Return page's effective /TrimBox, in PDF units.

        According to the PDF specification:
        "The trim box defines the intended dimensions of the finished page
        after trimming. It may be smaller than the media box to allow for
        production-related content, such as printing instructions, cut marks,
        or color bars."

        If the /TrimBox is not defined, the /CropBox is returned (and if
        /CropBox is not defined, /MediaBox is returned).
        """
    @trimbox.setter
    def trimbox(self, val: Array | Rectangle) -> None: ...
    @property
    def resources(self) -> Dictionary:
        """Return this page's resources dictionary.

        .. versionchanged:: 7.0.0
            If the resources dictionary does not exist, an empty one will be created.
            A TypeError is raised if a page has a /Resources key but it is not a
            dictionary.
        """
    def add_resource(
        self,
        res: Object,
        res_type: Name,
        name: Name | None = None,
        *,
        prefix: str = '',
        replace_existing: bool = True,
    ) -> Name:
        """Add a new resource to the page's Resources dictionary.

        If the Resources dictionaries do not exist, they will be created.

        Args:
            self: The object to add to the resources dictionary.
            res: The dictionary object to insert into the resources
                dictionary.
            res_type: Should be one of the following Resource dictionary types:
                ExtGState, ColorSpace, Pattern, Shading, XObject, Font, Properties.
            name: The name of the object. If omitted, a random name will be
                generated with enough randomness to be globally unique.
            prefix: A prefix for the name of the object. Allows conveniently
                namespacing when using random names, e.g. prefix="Im" for images.
                Mutually exclusive with name parameter.
            replace_existing: If the name already exists in one of the resource
                dictionaries, remove it.

        Example:
            >>> pdf = pikepdf.Pdf.new()
            >>> pdf.add_blank_page(page_size=(100, 100))
            <pikepdf.Page({
              "/Contents": pikepdf.Stream(owner=<...>, data=<...>, {
            <BLANKLINE>
              }),
              "/MediaBox": [ 0, 0, 100, 100 ],
              "/Parent": <reference to /Pages>,
              "/Resources": {
            <BLANKLINE>
              },
              "/Type": "/Page"
            })>
            >>> formxobj = pikepdf.Dictionary(
            ...     Type=Name.XObject,
            ...     Subtype=Name.Form
            ... )
            >>> resource_name = pdf.pages[0].add_resource(formxobj, Name.XObject)

        .. versionadded:: 2.3

        .. versionchanged:: 2.14
            If *res* does not belong to the same `Pdf` that owns this page,
            a copy of *res* is automatically created and added instead. In previous
            versions, it was necessary to change for this case manually.

        .. versionchanged:: 4.3.0
            Returns the name of the overlay in the resources dictionary instead
            of returning None.
        """

class PageList:
    """For accessing pages in a PDF.

    A ``list``-like object enumerating a range of pages in a :class:`pikepdf.Pdf`.
    It may be all of the pages or a subset. Obtain using :attr:`pikepdf.Pdf.pages`.

    See :class:`pikepdf.Page` for accessing individual pages.
    """

    def append(self, page: Page) -> None:
        """Add another page to the end.

        While this method copies pages from one document to another, it does not
        copy certain metadata such as annotations, form fields, bookmarks or
        structural tree elements. Copying these is a more complex, application
        specific operation.
        """
    def extend(self, other: PageList | Iterable[Page]) -> None:
        """Extend the ``Pdf`` by adding pages from an iterable of pages.

        While this method copies pages from one document to another, it does not
        copy certain metadata such as annotations, form fields, bookmarks or
        structural tree elements. Copying these is a more complex, application
        specific operation.
        """
    @overload
    def from_objgen(self, objgen: tuple[int, int]) -> Page: ...
    @overload
    def from_objgen(self, objgen: int, gen: int) -> Page: ...
    def from_objgen(
        self, objgen: tuple[int, int] | int, gen: int | None = None
    ) -> Page:
        """Given an objgen (object ID, generation), return the page.

        Raises an exception if no page matches.
        """
    def index(self, page: Page) -> int:
        """Given a page, find the index.

        That is, returns ``n`` such that ``pdf.pages[n] == this_page``.
        A ``ValueError`` exception is thrown if the page does not belong to
        to this ``Pdf``. The first page has index 0.
        """
    def insert(self, index: int, obj: Page) -> None:
        """Insert a page at the specified location.

        Args:
            index: location at which to insert page, 0-based indexing
            obj: page object to insert
        """
    def p(self, pnum: int) -> Page:
        """Look up page number in ordinal numbering, where 1 is the first page.

        This is provided for convenience in situations where ordinal numbering
        is more natural. It is equivalent to ``.pages[pnum - 1]``. ``.p(0)``
        is an error and negative indexing is not supported.

        If the PDF defines custom page labels (such as labeling front matter
        with Roman numerals and the main body with Arabic numerals), this
        function does not account for that. Use :attr:`pikepdf.Page.label`
        to get the page label for a page.
        """
    def remove(self, page: Page | None = None, *, p: int) -> None:
        """Remove a page.

        Args:
            page: If page is not None, remove that page.
            p: 1-based page number to remove, if page is None.
        """
    def reverse(self) -> None:
        """Reverse the order of pages."""
    @overload
    def __delitem__(self, idx: int) -> None: ...
    @overload
    def __delitem__(self, sl: slice) -> None: ...
    @overload
    def __getitem__(self, idx: int) -> Page: ...
    @overload
    def __getitem__(self, sl: slice) -> list[Page]: ...
    def __iter__(self) -> _PageListIterator: ...
    def __len__(self) -> int: ...
    @overload
    def __setitem__(self, idx: int, page: Page) -> None: ...
    @overload
    def __setitem__(self, sl: slice, pages: Iterable[Page]) -> None: ...

class _PageListIterator:
    def __iter__(self) -> _PageListIterator: ...
    def __next__(self) -> Page: ...

class Pdf:
    def _repr_mimebundle_(include: Any = ..., exclude: Any = ...) -> Any:
        """Present options to IPython or Jupyter for rich display of this object.

        See:
        https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display
        """
    def add_blank_page(self, *, page_size: tuple[Numeric, Numeric] = ...) -> Page:
        """Add a blank page to this PDF.

        If pages already exist, the page will be added to the end. Pages may be
        reordered using ``Pdf.pages``.

        The caller may add content to the page by modifying its objects after creating
        it.

        Args:
            page_size (tuple): The size of the page in PDF units (1/72 inch or 0.35mm).
                Default size is set to a US Letter 8.5" x 11" page.
        """
    def __enter__(self) -> Pdf: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _add_page(self, page: Object, first: bool = ...) -> None:
        """Low-level private method to attach a page to this PDF.

        The page can be either be a newly constructed PDF object or it can
        be obtained from another PDF.

        Args:
            page: The page object to attach.
            first: If True, prepend this before the first page;
                if False append after last page.
        """
    def _decode_all_streams_and_discard(self) -> None: ...
    def _get_object_id(self, arg0: int, arg1: int) -> Object: ...
    def _process(self, arg0: str, arg1: bytes) -> None: ...
    def _remove_page(self, arg0: Object) -> None: ...
    def _replace_object(self, arg0: tuple[int, int], arg1: Object) -> None: ...
    def _swap_objects(self, arg0: tuple[int, int], arg1: tuple[int, int]) -> None: ...
    def check(self) -> list[str]:
        """Check if PDF is syntactically well-formed.

        Similar to ``qpdf --check``, checks for syntax
        or structural problems in the PDF. This is mainly useful to PDF
        developers and may not be informative to the average user. PDFs with
        these problems still render correctly, if PDF viewers are capable of
        working around the issues they contain. In many cases, pikepdf can
        also fix the problems.

        An example problem found by this function is a xref table that is
        missing an object reference. A page dictionary with the wrong type of
        key, such as a string instead of an array of integers for its mediabox,
        is not the sort of issue checked for. If this were an XML checker, it
        would tell you if the XML is well-formed, but could not tell you if
        the XML is valid XHTML or if it can be rendered as a usable web page.

        This function also attempts to decompress all streams in the PDF.
        If no JBIG2 decoder is available and JBIG2 images are presented,
        a warning will occur that JBIG2 cannot be checked.

        This function returns a list of strings describing the issues. The
        text is subject to change and should not be treated as a stable API.

        Returns:
            Empty list if no issues were found. List of issues as text strings
            if issues were found.
        """
    def check_linearization(self, stream: object = ...) -> bool:
        """Reports information on the PDF's linearization.

        Args:
            stream: A stream to write this information too; must
                implement ``.write()`` and ``.flush()`` method. Defaults to
                :data:`sys.stderr`.

        Returns:
            ``True`` if the file is correctly linearized, and ``False`` if
            the file is linearized but the linearization data contains errors
            or was incorrectly generated.

        Raises:
            RuntimeError: If the PDF in question is not linearized at all.
        """
    def close(self) -> None:
        """Close a ``Pdf`` object and release resources acquired by pikepdf.

        If pikepdf opened the file handle it will close it (e.g. when opened with a file
        path). If the caller opened the file for pikepdf, the caller close the file.
        ``with`` blocks will call close when exit.

        pikepdf lazily loads data from PDFs, so some :class:`pikepdf.Object` may
        implicitly depend on the :class:`pikepdf.Pdf` being open. This is always the
        case for :class:`pikepdf.Stream` but can be true for any object. Do not close
        the `Pdf` object if you might still be accessing content from it.

        When an ``Object`` is copied from one ``Pdf`` to another, the ``Object`` is
        copied into the destination ``Pdf`` immediately, so after accessing all desired
        information from the source ``Pdf`` it may be closed.

        .. versionchanged:: 3.0
            In pikepdf 2.x, this function actually worked by resetting to a very short
            empty PDF. Code that relied on this quirk may not function correctly.
        """
    def copy_foreign(self, h: Object) -> Object:
        """Copy an ``Object`` from a foreign ``Pdf`` and return a copy.

        The object must be owned by a different ``Pdf`` from this one.

        If the object has previously been copied, return a reference to
        the existing copy, even if that copy has been modified in the meantime.

        If you want to copy a page from one PDF to another, use:
        ``pdf_b.pages[0] = pdf_a.pages[0]``. That interface accounts for the
        complexity of copying pages.

        This function is used to copy a :class:`pikepdf.Object` that is owned by
        some other ``Pdf`` into this one. This is performs a deep (recursive) copy
        and preserves all references that may exist in the foreign object. For
        example, if

            >>> object_a = pdf.copy_foreign(object_x)  # doctest: +SKIP
            >>> object_b = pdf.copy_foreign(object_y)  # doctest: +SKIP
            >>> object_c = pdf.copy_foreign(object_z)  # doctest: +SKIP

        and ``object_z`` is a shared descendant of both ``object_x`` and ``object_y``
        in the foreign PDF, then ``object_c`` is a shared descendant of both
        ``object_a`` and ``object_b`` in this PDF. If ``object_x`` and ``object_y``
        refer to the same object, then ``object_a`` and ``object_b`` are the
        same object.

        It also copies all :class:`pikepdf.Stream` objects. Since this may copy
        a large amount of data, it is not done implicitly. This function does
        not copy references to pages in the foreign PDF - it stops at page
        boundaries. Thus, if you use ``copy_foreign()`` on a table of contents
        (``/Outlines`` dictionary), you may have to update references to pages.

        Direct objects, including dictionaries, do not need ``copy_foreign()``.
        pikepdf will automatically convert and construct them.

        Note:
            pikepdf automatically treats incoming pages from a foreign PDF as
            foreign objects, so :attr:`Pdf.pages` does not require this treatment.

        See Also:
            `QPDF::copyForeignObject <https://qpdf.readthedocs.io/en/stable/design.html#copying-objects-from-other-pdf-files>`_

        .. versionchanged:: 2.1
            Error messages improved.
        """
    @overload
    def get_object(self, objgen: tuple[int, int]) -> Object: ...
    @overload
    def get_object(self, objgen: int, gen: int) -> Object: ...
    def get_object(
        self, objgen: tuple[int, int] | int, gen: int | None = None
    ) -> Object:
        """Retrieve an object from the PDF.

        Can be called with either a 2-tuple of (objid, gen) or
        two integers objid and gen.
        """
    def get_warnings(self) -> list: ...
    @overload
    def make_indirect(self, obj: T) -> T: ...
    def make_indirect(self, obj: Any) -> Object:
        """Attach an object to the Pdf as an indirect object.

        Direct objects appear inline in the binary encoding of the PDF.
        Indirect objects appear inline as references (in English, "look
        up object 4 generation 0") and then read from another location in
        the file. The PDF specification requires that certain objects
        are indirect - consult the PDF specification to confirm.

        Generally a resource that is shared should be attached as an
        indirect object. :class:`pikepdf.Stream` objects are always
        indirect, and creating them will automatically attach it to the
        Pdf.

        Args:
            obj: The object to attach. If this a :class:`pikepdf.Object`,
                it will be attached as an indirect object. If it is
                any other Python object, we attempt conversion to
                :class:`pikepdf.Object` attach the result. If the
                object is already an indirect object, a reference to
                the existing object is returned. If the ``pikepdf.Object``
                is owned by a different Pdf, an exception is raised; use
                :meth:`pikepdf.Object.copy_foreign` instead.

        See Also:
            :meth:`pikepdf.Object.is_indirect`
        """
    def make_stream(self, data: bytes, d=None, **kwargs) -> Stream:
        """Create a new pikepdf.Stream object that is attached to this PDF.

        See:
            :meth:`pikepdf.Stream.__new__`
        """
    @classmethod
    def new(cls) -> Pdf:
        """Create a new, empty PDF.

        This is best when you are constructing a PDF from scratch.

        In most cases, if you are working from an existing PDF, you should open the
        PDF using :meth:`pikepdf.Pdf.open` and transform it, instead of a creating
        a new one, to preserve metadata and structural information. For example,
        if you want to split a PDF into two parts, you should open the PDF and
        transform it into the desired parts, rather than creating a new PDF and
        copying pages into it.
        """
    @staticmethod
    def open(
        filename_or_stream: Path | str | BinaryIO,
        *,
        password: str | bytes = '',
        hex_password: bool = False,
        ignore_xref_streams: bool = False,
        suppress_warnings: bool = True,
        attempt_recovery: bool = True,
        inherit_page_attributes: bool = True,
        access_mode: AccessMode = AccessMode.default,
        allow_overwriting_input: bool = False,
    ) -> Pdf:
        """Open an existing file at *filename_or_stream*.

        If *filename_or_stream* is path-like, the file will be opened for reading. The
        file should not be modified by another process while it is open in pikepdf, or
        undefined behavior may occur. This is because the file may be lazily loaded.
        When ``.close()`` is called, the file handle that pikepdf opened will be closed.

        If *filename_or_stream* is stream, the data will be accessed as a readable
        binary stream, from the current position in that stream.  When ``pdf =
        Pdf.open(stream)`` is called on a stream, pikepdf will not call
        ``stream.close()``; the caller must call both ``pdf.close()`` and
        ``stream.close()``, in that order, when the Pdf and stream are no longer needed.
        Use with-blocks will call ``.close()`` automatically.

        Whether a file or stream is opened, you must ensure that the data is not
        modified by another thread or process, or undefined behavior will occur. You
        also may not overwrite the input file using ``.save()``, unless
        ``allow_overwriting_input=True``. This is because data may be lazily loaded.

        If you intend to edit the file in place, or want to protect the file against
        modification by another process, use ``allow_overwriting_input=True``. This
        tells pikepdf to make a private copy of the file.

        Any changes to the file must be persisted by using ``.save()``.

        Examples:
            >>> with Pdf.open("test.pdf") as pdf:  # doctest: +SKIP
            ...     pass

            >>> pdf = Pdf.open("test.pdf", password="rosebud")  # doctest: +SKIP

        Args:
            filename_or_stream: Filename or Python readable and seekable file
                stream of PDF to open.
            password: User or owner password to open an
                encrypted PDF. If the type of this parameter is ``str`` it will be
                encoded as UTF-8. If the type is ``bytes`` it will be saved verbatim.
                Passwords are always padded or truncated to 32 bytes internally. Use
                ASCII passwords for maximum compatibility.
            hex_password: If True, interpret the password as a
                hex-encoded version of the exact encryption key to use, without
                performing the normal key computation. Useful in forensics.
            ignore_xref_streams: If True, ignore cross-reference
                streams. See qpdf documentation.
            suppress_warnings: If True (default), warnings are not
                printed to stderr. Use :meth:`pikepdf.Pdf.get_warnings()` to retrieve
                warnings.
            attempt_recovery: If True (default), attempt to recover
                from PDF parsing errors.
            inherit_page_attributes: If True (default), push attributes
                set on a group of pages to individual pages
            access_mode: If ``.default``, pikepdf will
                decide how to access the file. Currently, it will always selected stream
                access. To attempt memory mapping and fallback to stream if memory
                mapping failed, use ``.mmap``.  Use ``.mmap_only`` to require memory
                mapping or fail (this is expected to only be useful for testing).
                Applications should be prepared to handle the SIGBUS signal on POSIX in
                the event that the file is successfully mapped but later goes away.
            allow_overwriting_input: If True, allows calling ``.save()``
                to overwrite the input file. This is performed by loading the entire
                input file into memory at open time; this will use more memory and may
                recent performance especially when the opened file will not be modified.

        Raises:
            pikepdf.PasswordError: If the password failed to open the
                file.
            pikepdf.PdfError: If for other reasons we could not open
                the file.
            TypeError: If the type of ``filename_or_stream`` is not
                usable.
            FileNotFoundError: If the file was not found.

        Note:
            When *filename_or_stream* is a stream and the stream is located on a
            network, pikepdf assumes that the stream using buffering and read caches to
            achieve reasonable performance. Streams that fetch data over a network in
            response to every read or seek request, no matter how small, will perform
            poorly. It may be easier to download a PDF from network to temporary local
            storage (such as ``io.BytesIO``), manipulate it, and then re-upload it.

        .. versionchanged:: 3.0
            Keyword arguments now mandatory for everything except the first
            argument.
        """
    def open_metadata(
        self,
        set_pikepdf_as_editor: bool = True,
        update_docinfo: bool = True,
        strict: bool = False,
    ) -> PdfMetadata:
        """Open the PDF's XMP metadata for editing.

        There is no ``.close()`` function on the metadata object, since this is
        intended to be used inside a ``with`` block only.

        For historical reasons, certain parts of PDF metadata are stored in
        two different locations and formats. This feature coordinates edits so
        that both types of metadata are updated consistently and "atomically"
        (assuming single threaded access). It operates on the ``Pdf`` in memory,
        not any file on disk. To persist metadata changes, you must still use
        ``Pdf.save()``.

        Example:
            >>> pdf = pikepdf.Pdf.open("../tests/resources/graph.pdf")
            >>> with pdf.open_metadata() as meta:
            ...     meta['dc:title'] = 'Set the Dublic Core Title'
            ...     meta['dc:description'] = 'Put the Abstract here'

        Args:
            set_pikepdf_as_editor: Automatically update the metadata ``pdf:Producer``
                to show that this version of pikepdf is the most recent software to
                modify the metadata, and ``xmp:MetadataDate`` to timestamp the update.
                Recommended, except for testing.

            update_docinfo: Update the standard fields of DocumentInfo
                (the old PDF metadata dictionary) to match the corresponding
                XMP fields. The mapping is described in
                :attr:`PdfMetadata.DOCINFO_MAPPING`. Nonstandard DocumentInfo
                fields and XMP metadata fields with no DocumentInfo equivalent
                are ignored.

            strict: If ``False`` (the default), we aggressively attempt
                to recover from any parse errors in XMP, and if that fails we
                overwrite the XMP with an empty XMP record.  If ``True``, raise
                errors when either metadata bytes are not valid and well-formed
                XMP (and thus, XML). Some trivial cases that are equivalent to
                empty or incomplete "XMP skeletons" are never treated as errors,
                and always replaced with a proper empty XMP block. Certain
                errors may be logged.
        """
    def open_outline(self, max_depth: int = 15, strict: bool = False) -> Outline:
        """Open the PDF outline ("bookmarks") for editing.

        Recommend for use in a ``with`` block. Changes are committed to the
        PDF when the block exits. (The ``Pdf`` must still be opened.)

        Example:
            >>> pdf = pikepdf.open('../tests/resources/outlines.pdf')
            >>> with pdf.open_outline() as outline:
            ...     outline.root.insert(0, pikepdf.OutlineItem('Intro', 0))

        Args:
            max_depth: Maximum recursion depth of the outline to be
                imported and re-written to the document. ``0`` means only
                considering the root level, ``1`` the first-level
                sub-outline of each root element, and so on. Items beyond
                this depth will be silently ignored. Default is ``15``.
            strict: With the default behavior (set to ``False``),
                structural errors (e.g. reference loops) in the PDF document
                will only cancel processing further nodes on that particular
                level, recovering the valid parts of the document outline
                without raising an exception. When set to ``True``, any such
                error will raise an ``OutlineStructureError``, leaving the
                invalid parts in place.
                Similarly, outline objects that have been accidentally
                duplicated in the ``Outline`` container will be silently
                fixed (i.e. reproduced as new objects) or raise an
                ``OutlineStructureError``.
        """
    def remove_unreferenced_resources(self) -> None:
        """Remove from /Resources any object not referenced in page's contents.

        PDF pages may share resource dictionaries with other pages. If
        pikepdf is used for page splitting, pages may reference resources
        in their /Resources dictionary that are not actually required.
        This purges all unnecessary resource entries.

        For clarity, if all references to any type of object are removed, that
        object will be excluded from the output PDF on save. (Conversely, only
        objects that are discoverable from the PDF's root object are included.)
        This function removes objects that are referenced from the page /Resources
        dictionary, but never called for in the content stream, making them
        unnecessary.

        Suggested before saving, if content streams or /Resources dictionaries
        are edited.
        """
    def save(
        self,
        filename_or_stream: Path | str | BinaryIO | None = None,
        *,
        static_id: bool = False,
        preserve_pdfa: bool = True,
        min_version: str | tuple[str, int] = '',
        force_version: str | tuple[str, int] = '',
        fix_metadata_version: bool = True,
        compress_streams: bool = True,
        stream_decode_level: StreamDecodeLevel | None = None,
        object_stream_mode: ObjectStreamMode = ObjectStreamMode.preserve,
        normalize_content: bool = False,
        linearize: bool = False,
        qdf: bool = False,
        progress: Callable[[int], None] | None = None,
        encryption: Encryption | bool | None = None,
        recompress_flate: bool = False,
        deterministic_id: bool = False,
    ) -> None:
        """Save all modifications to this :class:`pikepdf.Pdf`.

        Args:
            filename_or_stream: Where to write the output. If a file
                exists in this location it will be overwritten.
                If the file was opened with ``allow_overwriting_input=True``,
                then it is permitted to overwrite the original file, and
                this parameter may be omitted to implicitly use the original
                filename. Otherwise, the filename may not be the same as the
                input file, as overwriting the input file would corrupt data
                since pikepdf using lazy loading.

            static_id: Indicates that the ``/ID`` metadata, normally
                calculated as a hash of certain PDF contents and metadata
                including the current time, should instead be set to a static
                value. Only use this for debugging and testing. Use
                ``deterministic_id`` if you want to get the same ``/ID`` for
                the same document contents.
            preserve_pdfa: Ensures that the file is generated in a
                manner compliant with PDF/A and other stricter variants.
                This should be True, the default, in most cases.

            min_version: Sets the minimum version of PDF
                specification that should be required. If left alone qpdf
                will decide. If a tuple, the second element is an integer, the
                extension level. If the version number is not a valid format,
                qpdf will decide what to do.
            force_version: Override the version recommend by qpdf,
                potentially creating an invalid file that does not display
                in old versions. See qpdf manual for details. If a tuple, the
                second element is an integer, the extension level.
            fix_metadata_version: If ``True`` (default) and the XMP metadata
                contains the optional PDF version field, ensure the version in
                metadata is correct. If the XMP metadata does not contain a PDF
                version field, none will be added. To ensure that the field is
                added, edit the metadata and insert a placeholder value in
                ``pdf:PDFVersion``. If XMP metadata does not exist, it will
                not be created regardless of the value of this argument.

            object_stream_mode:
                ``disable`` prevents the use of object streams.
                ``preserve`` keeps object streams from the input file.
                ``generate`` uses object streams wherever possible,
                creating the smallest files but requiring PDF 1.5+.

            compress_streams: Enables or disables the compression of
                uncompressed stream objects. By default this is set to
                ``True``, and the only reason to set it to ``False`` is for
                debugging or inspecting PDF contents.

                When enabled, uncompressed stream objects will be compressed
                whether they were uncompressed in the PDF when it was opened,
                or when the user creates new :class:`pikepdf.Stream` objects
                attached to the PDF. Stream objects can also be created
                indirectly, such as when content from another PDF is merged
                into the one being saved.

                Only stream objects that have no compression will be
                compressed when this object is set. If the object is
                compressed, compression will be preserved.

                Setting compress_streams=False does not trigger decompression
                unless decompression is specifically requested by setting
                both ``compress_streams=False`` and ``stream_decode_level``
                to the desired decode level (e.g. ``.generalized`` will
                decompress most non-image content).

                This option does not trigger recompression of existing
                compressed streams. For that, use ``recompress_flate``.

                The XMP metadata stream object, if present, is never
                compressed, to facilitate metadata reading by parsers that
                don't understand the full structure of PDF.

            stream_decode_level: Specifies how
                to encode stream objects. See documentation for
                :class:`pikepdf.StreamDecodeLevel`.

            recompress_flate: When disabled (the default), qpdf does not
                uncompress and recompress streams compressed with the Flate
                compression algorithm. If True, pikepdf will instruct qpdf to
                do this, which may be useful if recompressing streams to a
                higher compression level.

            normalize_content: Enables parsing and reformatting the
                content stream within PDFs. This may debugging PDFs easier.

            linearize: Enables creating linear or "fast web view",
                where the file's contents are organized sequentially so that
                a viewer can begin rendering before it has the whole file.
                As a drawback, it tends to make files larger.

            qdf: Save output QDF mode.  QDF mode is a special output
                mode in qpdf to allow editing of PDFs in a text editor. Use
                the program ``fix-qdf`` to fix convert back to a standard
                PDF.

            progress: Specify a callback function that is called
                as the PDF is written. The function will be called with an
                integer between 0-100 as the sole parameter, the progress
                percentage. This function may not access or modify the PDF
                while it is being written, or data corruption will almost
                certainly occur.

            encryption: If ``False``
                or omitted, existing encryption will be removed. If ``True``
                encryption settings are copied from the originating PDF.
                Alternately, an ``Encryption`` object may be provided that
                sets the parameters for new encryption.

            deterministic_id: Indicates that the ``/ID`` metadata, normally
                calculated as a hash of certain PDF contents and metadata
                including the current time, should instead be computed using
                only deterministic data like the file contents. At a small
                runtime cost, this enables generation of the same ``/ID`` if
                the same inputs are converted in the same way multiple times.
                Does not work for encrypted files.

        Raises:
            PdfError
            ForeignObjectError
            ValueError

        You may call ``.save()`` multiple times with different parameters
        to generate different versions of a file, and you *may* continue
        to modify the file after saving it. ``.save()`` does not modify
        the ``Pdf`` object in memory, except possibly by updating the XMP
        metadata version with ``fix_metadata_version``.

        .. note::

            :meth:`pikepdf.Pdf.remove_unreferenced_resources` before saving
            may eliminate unnecessary resources from the output file if there
            are any objects (such as images) that are referenced in a page's
            Resources dictionary but never called in the page's content stream.

        .. note::

            pikepdf can read PDFs with incremental updates, but always
            coalesces any incremental updates into a single non-incremental
            PDF file when saving.

        .. note::
            If filename_or_stream is a stream and the process is interrupted during
            writing, the stream may be left in a corrupt state. It is the
            responsibility of the caller to manage the stream in this case.

        .. versionchanged:: 2.7
            Added *recompress_flate*.

        .. versionchanged:: 3.0
            Keyword arguments now mandatory for everything except the first
            argument.

        .. versionchanged:: 8.1
            If filename_or_stream is a filename and that file exists, the new file
            is written to a temporary file in the same directory and then moved into
            place. This prevents the existing destination file from being corrupted
            if the process is interrupted during writing; previously, corrupting the
            destination file was possible. If no file exists at the destination, output
            is written directly to the destination, but the destination will be deleted
            if errors occur during writing. Prior to 8.1, the file was always written
            directly to the destination, which could result in a corrupt destination
            file if the process was interrupted during writing.

        .. versionchanged:: 9.1
            When opened with ``allow_overwriting_input=True``, we now attempt to
            restore the original file permissions, ownership and creation time.
            The modified time is always set to the time of saving. An unusual
            umask or other settings changes still cause a failure to restore
            permissions.
        """
    def show_xref_table(self) -> None:
        """Pretty-print the Pdf's xref (cross-reference table)."""
    @property
    def Root(self) -> Object: ...
    @property
    def _allow_accessibility(self) -> bool: ...
    @property
    def _allow_extract(self) -> bool: ...
    @property
    def _allow_modify_all(self) -> bool: ...
    @property
    def _allow_modify_annotation(self) -> bool: ...
    @property
    def _allow_modify_assembly(self) -> bool: ...
    @property
    def _allow_modify_form(self) -> bool: ...
    @property
    def _allow_modify_other(self) -> bool: ...
    @property
    def _allow_print_highres(self) -> bool: ...
    @property
    def _allow_print_lowres(self) -> bool: ...
    @property
    def _encryption_data(self) -> dict: ...
    @property
    def _pages(self) -> Any: ...
    @property
    def allow(self) -> Permissions:
        """Report permissions associated with this PDF.

        By default these permissions will be replicated when the PDF is
        saved. Permissions may also only be changed when a PDF is being saved,
        and are only available for encrypted PDFs. If a PDF is not encrypted,
        all operations are reported as allowed.

        pikepdf has no way of enforcing permissions.
        """
    @property
    def docinfo(self) -> Object:
        """Access the (deprecated) document information dictionary.

        The document information dictionary is a brief metadata record that can
        store some information about the origin of a PDF. It is deprecated and
        removed in the PDF 2.0 specification (not deprecated from the
        perspective of pikepdf). Use the ``.open_metadata()`` API instead, which
        will edit the modern (and unfortunately, more complicated) XMP metadata
        object and synchronize changes to the document information dictionary.

        This property simplifies access to the actual document information
        dictionary and ensures that it is created correctly if it needs to be
        created.

        A new, empty dictionary will be created if this property is accessed
        and dictionary does not exist. (This is to ensure that convenient code
        like ``pdf.docinfo[Name.Title] = "Title"`` will work when the dictionary
        does not exist at all.)

        You can delete the document information dictionary by deleting this property,
        ``del pdf.docinfo``. Note that accessing the property after deleting it
        will re-create with a new, empty dictionary.

        .. versionchanged:: 2.4
            Added support for ``del pdf.docinfo``.
        """
    @docinfo.setter
    def docinfo(self, val: Object) -> None: ...
    @property
    def encryption(self) -> EncryptionInfo:
        """Report encryption information for this PDF.

        Encryption settings may only be changed when a PDF is saved.
        """
    @property
    def extension_level(self) -> int:
        """Returns the extension level of this PDF.

        If a developer has released multiple extensions of a PDF version against
        the same base version value, they shall increase the extension level
        by 1. To be interpreted with :attr:`pdf_version`.
        """
    @property
    def filename(self) -> str:
        """The source filename of an existing PDF, when available.

        When the Pdf was created from scratch, this returns 'empty PDF'.
        When the Pdf was created from a stream, the return value is the
        word 'stream' followed by some information about the stream, if
        available.
        """
    @property
    def is_encrypted(self) -> bool:
        """Returns True if the PDF is encrypted.

        For information about the nature of the encryption, see
        :attr:`Pdf.encryption`.
        """
    @property
    def is_linearized(self) -> bool:
        """Returns True if the PDF is linearized.

        Specifically returns True iff the file starts with a linearization
        parameter dictionary.  Does no additional validation.
        """
    @property
    def objects(self) -> _ObjectList:
        """Return an iterable list of all objects in the PDF.

        After deleting content from a PDF such as pages, objects related
        to that page, such as images on the page, may still be present in
        this list.
        """
    @property
    def pages(self) -> PageList:
        """Returns the list of pages."""
    @property
    def pdf_version(self) -> str:
        """The version of the PDF specification used for this file, such as '1.7'.

        More precise information about the PDF version can be opened from the
        Pdf's XMP metadata.
        """
    @property
    def root(self) -> Object:
        """The /Root object of the PDF."""
    @property
    def trailer(self) -> Object:
        """Provides access to the PDF trailer object.

        See |pdfrm| section 7.5.5. Generally speaking,
        the trailer should not be modified with pikepdf, and modifying it
        may not work. Some of the values in the trailer are automatically
        changed when a file is saved.
        """
    @property
    def user_password_matched(self) -> bool:
        """Returns True if the user password matched when the ``Pdf`` was opened.

        It is possible for both the user and owner passwords to match.

        .. versionadded:: 2.10
        """
    @property
    def owner_password_matched(self) -> bool:
        """Returns True if the owner password matched when the ``Pdf`` was opened.

        It is possible for both the user and owner passwords to match.

        .. versionadded:: 2.10
        """
    def generate_appearance_streams(self) -> None:
        """Generates appearance streams for AcroForm forms and form fields.

        Appearance streams describe exactly how annotations and form fields
        should appear to the user. If omitted, the PDF viewer is free to
        render the annotations and form fields according to its own settings,
        as needed.

        For every form field in the document, this generates appearance
        streams, subject to the limitations of qpdf's ability to create
        appearance streams.

        When invoked, this method will modify the ``Pdf`` in memory. It may be
        best to do this after the ``Pdf`` is opened, or before it is saved,
        because it may modify objects that the user does not expect to be
        modified.

        If ``Pdf.Root.AcroForm.NeedAppearances`` is ``False`` or not present, no
        action is taken (because no appearance streams need to be generated).
        If ``True``, the appearance streams are generated, and the NeedAppearances
        flag is set to ``False``.

        See:
            https://github.com/qpdf/qpdf/blob/bf6b9ba1c681a6fac6d585c6262fb2778d4bb9d2/include/qpdf/QPDFFormFieldObjectHelper.hh#L216

        .. versionadded:: 2.11
        """
    def flatten_annotations(self, mode: str) -> None:
        """Flattens all PDF annotations into regular PDF content.

        Annotations are markup such as review comments, highlights, proofreading
        marks. User data entered into interactive form fields also counts as an
        annotation.

        When annotations are flattened, they are "burned into" the regular
        content stream of the document and the fact that they were once annotations
        is deleted. This can be useful when preparing a document for printing,
        to ensure annotations are printed, or to finalize a form that should
        no longer be changed.

        Args:
            mode: One of the strings ``'all'``, ``'screen'``, ``'print'``. If
                omitted or  set to empty, treated as ``'all'``. ``'screen'``
                flattens all except those marked with the PDF flag /NoView.
                ``'print'`` flattens only those marked for printing.
                Default is ``'all'``.

        .. versionadded:: 2.11
        """
    @property
    def attachments(self) -> Attachments:
        """Returns a mapping that provides access to all files attached to this PDF.

        PDF supports attaching (or embedding, if you prefer) any other type of file,
        including other PDFs. This property provides read and write access to
        these objects by filename.
        """

class Rectangle:
    """A PDF rectangle.

    Typically this will be a rectangle in PDF units (points, 1/72").
    Unlike raster graphics, the rectangle is defined by the **lower**
    left and upper right points.

    Rectangles in PDF are encoded as :class:`pikepdf.Array` with exactly
    four numeric elements, ordered as ``llx lly urx ury``.
    See |pdfrm| section 7.9.5.

    The rectangle may be considered degenerate if the lower left corner
    is not strictly less than the upper right corner.

    .. versionadded:: 2.14

    .. versionchanged:: 8.5
        Added operators to test whether rectangle ``a`` is contained in
        rectangle ``b`` (``a <= b``) and to calculate their intersection
        (``a & b``).
    """

    llx: float = ...
    """The lower left corner on the x-axis."""
    lly: float = ...
    """The lower left corner on the y-axis."""
    urx: float = ...
    """The upper right corner on the x-axis."""
    ury: float = ...
    """The upper right corner on the y-axis."""
    @overload
    def __init__(self, llx: float, lly: float, urx: float, ury: float, /) -> None: ...
    @overload
    def __init__(self, other: Rectangle) -> None: ...
    @overload
    def __init__(self, other: Array) -> None: ...
    def __init__(self, *args) -> None:
        """Construct a new rectangle."""
    def __and__(self, other: Rectangle) -> Rectangle:
        """Return the bounding Rectangle of the common area of self and other."""
    def __le__(self, other: Rectangle) -> bool:
        """Return True if self is contained in other or equal to other."""
    @property
    def width(self) -> float:
        """The width of the rectangle."""
    @property
    def height(self) -> float:
        """The height of the rectangle."""
    @property
    def lower_left(self) -> tuple[float, float]:
        """A point for the lower left corner."""
    @property
    def lower_right(self) -> tuple[float, float]:
        """A point for the lower right corner."""
    @property
    def upper_left(self) -> tuple[float, float]:
        """A point for the upper left corner."""
    @property
    def upper_right(self) -> tuple[float, float]:
        """A point for the upper right corner."""
    def as_array(self) -> Array:
        """Returns this rectangle as a :class:`pikepdf.Array`."""
    def __eq__(self, other: Any) -> bool: ...
    def __repr__(self) -> str: ...

class NameTree(MutableMapping[str | bytes, Object]):
    """An object for managing *name tree* data structures in PDFs.

    A name tree is a key-value data structure. The keys are any binary strings
    (that is, Python ``bytes``). If ``str`` selected is provided as a key,
    the UTF-8 encoding of that string is tested. Name trees are (confusingly)
    not indexed by ``pikepdf.Name`` objects. They behave like
    ``DictMapping[bytes, pikepdf.Object]``.

    The keys are sorted; pikepdf will ensure that the order is preserved.

    The value may be any PDF object. Typically it will be a dictionary or array.

    Internally in the PDF, a name tree can be a fairly complex tree data structure
    implemented with many dictionaries and arrays. pikepdf (using libqpdf)
    will automatically read, repair and maintain this tree for you. There should not
    be any reason to access the internal nodes of a number tree; use this
    interface instead.

    NameTrees are used to store certain objects like file attachments in a PDF.
    Where a more specific interface exists, use that instead, and it will
    manipulate the name tree in a semantic correct manner for you.

    Do not modify the internal structure of a name tree while you have a
    ``NameTree`` referencing it. Access it only through the ``NameTree`` object.

    Names trees are described in the |pdfrm| section 7.9.6. See section 7.7.4
    for a list of PDF objects that are stored in name trees.

    .. versionadded:: 3.0
    """

    @staticmethod
    def new(pdf: Pdf, *, auto_repair: bool = True) -> NameTree:
        """Create a new NameTree in the provided Pdf.

        You will probably need to insert the name tree in the PDF's
        catalog. For example, to insert this name tree in
        /Root /Names /Dests:

        .. code-block:: python

            nt = NameTree.new(pdf)
            pdf.Root.Names.Dests = nt.obj
        """
    def __contains__(self, name: object) -> bool: ...
    def __delitem__(self, name: str | bytes) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __getitem__(self, name: str | bytes) -> Object: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def __len__(self) -> int: ...
    def __setitem__(self, name: str | bytes, o: Object) -> None: ...
    def __init__(self, obj: Object, *, auto_repair: bool = ...) -> None: ...
    def _as_map(self) -> _ObjectMapping: ...
    @property
    def obj(self) -> Object:
        """Returns the underlying root object for this name tree."""

class NumberTree(MutableMapping[int, Object]):
    """An object for managing *number tree* data structures in PDFs.

    A number tree is a key-value data structure, like name trees, except that the
    key is an integer. It behaves like ``Dict[int, pikepdf.Object]``.

    The keys can be sparse - not all integers positions will be populated. Keys
    are also always sorted; pikepdf will ensure that the order is preserved.

    The value may be any PDF object. Typically it will be a dictionary or array.

    Internally in the PDF, a number tree can be a fairly complex tree data structure
    implemented with many dictionaries and arrays. pikepdf (using libqpdf)
    will automatically read, repair and maintain this tree for you. There should not
    be any reason to access the internal nodes of a number tree; use this
    interface instead.

    NumberTrees are not used much in PDF. The main thing they provide is a mapping
    between 0-based page numbers and user-facing page numbers (which pikepdf
    also exposes as ``Page.label``). The ``/PageLabels`` number tree is where the
    page numbering rules are defined.

    Number trees are described in the |pdfrm| section 7.9.7. See section 12.4.2
    for a description of the page labels number tree. Here is an example of modifying
    an existing page labels number tree:

    .. code-block:: python

        pagelabels = NumberTree(pdf.Root.PageLabels)
        # Label pages starting at 0 with lowercase Roman numerals
        pagelabels[0] = Dictionary(S=Name.r)
        # Label pages starting at 6 with decimal numbers
        pagelabels[6] = Dictionary(S=Name.D)

        # Page labels will now be:
        # i, ii, iii, iv, v, 1, 2, 3, ...

    Do not modify the internal structure of a name tree while you have a
    ``NumberTree`` referencing it. Access it only through the ``NumberTree`` object.

    .. versionadded:: 5.4
    """

    @staticmethod
    def new(pdf: Pdf, *, auto_repair: bool = True) -> NumberTree:
        """Create a new NumberTree in the provided Pdf.

        You will probably need to insert the number tree in the PDF's
        catalog. For example, to insert this number tree in
        /Root /PageLabels:

        .. code-block:: python

            nt = NumberTree.new(pdf)
            pdf.Root.PageLabels = nt.obj
        """
    def __contains__(self, key: object) -> bool: ...
    def __delitem__(self, key: int) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __getitem__(self, key: int) -> Object: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: int, o: Object) -> None: ...
    def __init__(self, obj: Object, *, auto_repair: bool = ...) -> None: ...
    def _as_map(self) -> _ObjectMapping: ...
    @property
    def obj(self) -> Object: ...

class ContentStreamInstruction:
    """Represents one complete instruction inside a content stream."""

    @overload
    def __init__(self, operands: _ObjectList, operator: Operator) -> None: ...
    @overload
    def __init__(
        self, operands: Iterable[Object | int | float | Array], operator: Operator
    ) -> None: ...
    @overload
    def __init__(self, other: ContentStreamInstruction) -> None: ...
    def __init__(self, *args) -> None: ...
    @property
    def operands(self) -> _ObjectList: ...
    @property
    def operator(self) -> Operator: ...
    def __getitem__(self, index: int) -> _ObjectList | Operator: ...
    def __len__(self) -> int: ...

class ContentStreamInlineImage:
    """Represents an instruction to draw an inline image.

    pikepdf consolidates the BI-ID-EI sequence of operators, as appears in a PDF to
    declare an inline image, and replaces them with a single virtual content stream
    instruction with the operator "INLINE IMAGE".
    """

    @property
    def operands(self) -> _ObjectList: ...
    @property
    def operator(self) -> Operator: ...
    def __getitem__(self, index: int) -> _ObjectList | Operator: ...
    def __len__(self) -> int: ...
    @property
    def iimage(self) -> PdfInlineImage: ...

class Job:
    """Provides access to the qpdf job interface.

    All of the functionality of the ``qpdf`` command line program
    is now available to pikepdf through jobs.

    For further details:
        https://qpdf.readthedocs.io/en/stable/qpdf-job.html
    """

    EXIT_ERROR: ClassVar[int] = 2
    """Exit code for a job that had an error."""
    EXIT_WARNING: ClassVar[int] = 3
    """Exit code for a job that had a warning."""
    EXIT_IS_NOT_ENCRYPTED: ClassVar[int] = 2
    """Exit code for a job that provide a password when the input was not encrypted."""
    EXIT_CORRECT_PASSWORD: ClassVar[int] = 3
    LATEST_JOB_JSON: ClassVar[int]
    """Version number of the most recent job-JSON schema."""
    LATEST_JSON: ClassVar[int]
    """Version number of the most recent qpdf-JSON schema."""

    @staticmethod
    def json_out_schema(*, schema: int) -> str:
        """For reference, the qpdf JSON output schema is built-in."""
    @staticmethod
    def job_json_schema(*, schema: int) -> str:
        """For reference, the qpdf job command line schema is built-in."""
    @overload
    def __init__(self, json: str) -> None: ...
    @overload
    def __init__(self, json_dict: Mapping) -> None: ...
    @overload
    def __init__(
        self, args: Sequence[str | bytes], *, progname: str = 'pikepdf'
    ) -> None: ...
    def __init__(self, *args, **kwargs) -> None:
        """Create a Job from command line arguments to the qpdf program.

        The first item in the ``args`` list should be equal to ``progname``,
        whose default is ``"pikepdf"``.

        Example:
            job = Job(['pikepdf', '--check', 'input.pdf'])
            job.run()
        """
    def check_configuration(self) -> None:
        """Checks if the configuration is valid; raises an exception if not."""
    @property
    def creates_output(self) -> bool:
        """Returns True if the Job will create some sort of output file."""
    @property
    def message_prefix(self) -> str:
        """Allows manipulation of the prefix in front of all output messages."""
    def run(self) -> None:
        """Executes the job."""
    def create_pdf(self):
        """Executes the first stage of the job."""
    def write_pdf(self, pdf: Pdf):
        """Executes the second stage of the job."""
    @property
    def has_warnings(self) -> bool:
        """After run(), returns True if there were warnings."""
    @property
    def exit_code(self) -> int:
        """After run(), returns an integer exit code.

        The meaning of exit code depends on the details of the Job that was run.
        Details are subject to change in libqpdf. Use properties ``has_warnings``
        and ``encryption_status`` instead.
        """
    @property
    def encryption_status(self) -> dict[str, bool]:
        """Returns a Python dictionary describing the encryption status."""

class Matrix:
    r"""A 2D affine matrix for PDF transformations.

    PDF uses matrices to transform document coordinates to screen/device
    coordinates.

    PDF matrices are encoded as :class:`pikepdf.Array` with exactly
    six numeric elements, ordered as ``a b c d e f``.

    .. math::

        \begin{bmatrix}
        a & b & 0 \\
        c & d & 0 \\
        e & f & 1 \\
        \end{bmatrix}

    The approximate interpretation of these six parameters is documented
    below. The values (0, 0, 1) in the third column are fixed, so a
    general 3×3 matrix cannot be converted to a PDF matrix.

    PDF transformation matrices are the transpose of most textbook
    treatments.  In a textbook, typically ``A × vc`` is used to
    transform a column vector ``vc=(x, y, 1)`` by the affine matrix ``A``.
    In PDF, the matrix is the transpose of that in the textbook,
    and ``vr × A'`` is used to transform a row vector ``vr=(x, y, 1)``.

    Transformation matrices specify the transformation from the new
    (transformed) coordinate system to the original (untransformed)
    coordinate system. x' and y' are the coordinates in the
    *untransformed* coordinate system, and x and y are the
    coordinates in the *transformed* coordinate system.

    PDF order:

    .. math::

        \begin{equation}
        \begin{bmatrix}
        x' & y' & 1
        \end{bmatrix}
        =
        \begin{bmatrix}
        x & y & 1
        \end{bmatrix}
        \begin{bmatrix}
        a & b & 0 \\
        c & d & 0 \\
        e & f & 1
        \end{bmatrix}
        \end{equation}

    To concatenate transformations, use the matrix multiple (``@``)
    operator to **pre**-multiply the next transformation onto existing
    transformations.

    Alternatively, use the .translated(), .scaled(), and .rotated()
    methods to chain transformation operations.

    Addition and other operations are not implemented because they're not
    that meaningful in a PDF context.

    Matrix objects are immutable. All transformation methods return
    new matrix objects.

    .. versionadded:: 8.7
    """

    @overload
    def __init__(self):
        """Construct an identity matrix."""
    @overload
    def __init__(
        self, a: float, b: float, c: float, d: float, e: float, f: float, /
    ): ...
    @overload
    def __init__(self, other: Matrix): ...
    @overload
    def __init__(self, values: tuple[float, float, float, float, float, float], /): ...
    @property
    def a(self) -> float:
        """``a`` is the horizontal scaling factor."""
    @property
    def b(self) -> float:
        """``b`` is horizontal skewing."""
    @property
    def c(self) -> float:
        """``c`` is vertical skewing."""
    @property
    def d(self) -> float:
        """``d`` is the vertical scaling factor."""
    @property
    def e(self) -> float:
        """``e`` is the horizontal translation."""
    @property
    def f(self) -> float:
        """``f`` is the vertical translation."""
    @property
    def shorthand(self) -> tuple[float, float, float, float, float, float]:
        """Return the 6-tuple (a,b,c,d,e,f) that describes this matrix."""
    def encode(self) -> bytes:
        """Encode matrix to bytes suitable for including in a PDF content stream."""
    def translated(self, tx, ty) -> Matrix:
        """Return a translated copy of this matrix.

        Calculates ``Matrix(1, 0, 0, 1, tx, ty) @ self``.

        Args:
            tx: horizontal translation
            ty: vertical translation
        """
    def scaled(self, sx, sy) -> Matrix:
        """Return a scaled copy of this matrix.

        Calculates ``Matrix(sx, 0, 0, sy, 0, 0) @ self``.

        Args:
            sx: horizontal scaling
            sy: vertical scaling
        """
    def rotated(self, angle_degrees_ccw) -> Matrix:
        """Return a rotated copy of this matrix.

        Calculates
        ``Matrix(cos(angle), sin(angle), -sin(angle), cos(angle), 0, 0) @ self``.

        Args:
            angle_degrees_ccw: angle in degrees counterclockwise
        """
    def __matmul__(self, other: Matrix) -> Matrix:
        """Return the matrix product of two matrices.

        Can be used to concatenate transformations. Transformations should be
        composed by **pre**-multiplying matrices. For example, to apply a
        scaling transform, one could do::

            scale = pikepdf.Matrix(2, 0, 0, 2, 0, 0)
            scaled = scale @ matrix
        """
    def inverse(self) -> Matrix:
        """Return the inverse of the matrix.

        The inverse matrix reverses the transformation of the original matrix.

        In rare situations, the inverse may not exist. In that case, an
        exception is thrown. The PDF will likely have rendering problems.
        """
    def __array__(self, dtype: Any = None, copy: bool | None = True) -> np.ndarray:
        """Convert this matrix to a NumPy array of type dtype.

        If copy is True, a copy is made. If copy is False, an exception is raised.

        If numpy is not installed, this will throw an exception.
        """
    def as_array(self) -> Array:
        """Convert this matrix to a pikepdf.Array.

        A Matrix cannot be inserted into a PDF directly. Use this function
        to convert a Matrix to a pikepdf.Array, which can be inserted.
        """
    @overload
    def transform(self, point: tuple[float, float]) -> tuple[float, float]:
        """Transform a point by this matrix.

        Computes [x y 1] @ self.
        """
    @overload
    def transform(self, rect: Rectangle) -> Rectangle: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    def __getstate__(self) -> tuple[float, float, float, float, float, float]: ...
    def __setstate__(
        self, state: tuple[float, float, float, float, float, float]
    ) -> None: ...

def _Null() -> Any: ...
def _encode(handle: Any) -> Object: ...
def _new_array(arg0: Iterable) -> Array:
    """Low-level function to construct a PDF Array.

    Construct a PDF Array object from an iterable of PDF objects or types
    that can be coerced to PDF objects.
    """

def _new_boolean(arg0: bool) -> Object:
    """Low-level function to construct a PDF Boolean.

    pikepdf automatically converts PDF booleans to Python booleans and
    vice versa. This function serves no purpose other than to test
    that functionality.
    """

def _new_dictionary(arg0: Mapping[Any, Any]) -> Dictionary:
    """Low-level function to construct a PDF Dictionary.

    Construct a PDF Dictionary from a mapping of PDF objects or Python types
    that can be coerced to PDF objects."
    """

def _new_integer(arg0: int) -> int:
    """Low-level function to construct a PDF Integer.

    pikepdf automatically converts PDF integers to Python integers and
    vice versa. This function serves no purpose other than to test
    that functionality.
    """

def _new_name(s: str | bytes) -> Name:
    """Low-level function to construct a PDF Name.

    Must begin with '/'. Certain characters are escaped according to
    the PDF specification.
    """

def _new_operator(op: str) -> Operator:
    """Low-level function to construct a PDF Operator."""

@overload
def _new_real(s: str) -> Decimal:  # noqa: D418
    """Low-level function to construct a PDF Real.

    pikepdf automatically PDF real numbers to Python Decimals.
    This function serves no purpose other than to test that
    functionality.
    """

@overload
def _new_real(value: float, places: int = ...) -> Decimal:  # noqa: D418
    """Low-level function to construct a PDF Real.

    pikepdf automatically PDF real numbers to Python Decimals.
    This function serves no purpose other than to test that
    functionality.
    """

def _new_stream(owner: Pdf, data: bytes) -> Stream:
    """Low-level function to construct a PDF Stream.

    Construct a PDF Stream object from binary data.
    """

def _new_string(s: str | bytes) -> String:
    """Low-level function to construct a PDF String object."""

def _new_string_utf8(s: str) -> String:
    """Low-level function to construct a PDF String object from UTF-8 bytes."""

def _translate_qpdf_logic_error(arg0: str) -> str: ...
def get_decimal_precision() -> int:
    """Set the number of decimal digits to use when converting floats."""

def pdf_doc_to_utf8(pdfdoc: bytes) -> str:
    """Low-level function to convert PDFDocEncoding to UTF-8.

    Use the pdfdoc codec instead of using this directly.
    """

def qpdf_version() -> str: ...
def set_access_default_mmap(mmap: bool) -> bool: ...
def get_access_default_mmap() -> bool: ...
def set_decimal_precision(prec: int) -> int:
    """Get the number of decimal digits to use when converting floats."""

def unparse(obj: Any) -> bytes: ...
def utf8_to_pdf_doc(utf8: str, unknown: bytes) -> tuple[bool, bytes]: ...
def _unparse_content_stream(contentstream: Iterable[Any]) -> bytes: ...
def set_flate_compression_level(
    level: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
) -> int:
    """Set compression level whenever Flate compression is used.

    Args:
        level: -1 (default), 0 (no compression), 1 to 9 (increasing compression)
    """
