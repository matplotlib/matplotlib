# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Provide classes to stand in for PDF objects.

The purpose of these is to provide nice-looking classes to allow explicit
construction of PDF objects and more pythonic idioms and facilitate discovery
by documentation generators and linters.

It's also a place to narrow the scope of input types to those more easily
converted to C++.

There is some deliberate "smoke and mirrors" here: all of the objects are truly
instances of ``pikepdf.Object``, which is a variant container object. The
``__new__`` constructs a ``pikepdf.Object`` in each case, and the rest of the
class definition is present as an aide for code introspection.
"""

from __future__ import annotations

# pylint: disable=unused-import, abstract-method
from secrets import token_urlsafe
from typing import TYPE_CHECKING, Any, Iterable, Mapping, cast

from pikepdf import _core
from pikepdf._core import Matrix, Object, ObjectType, Rectangle

if TYPE_CHECKING:  # pragma: no cover
    from pikepdf import Pdf

# By default pikepdf.Object will identify itself as pikepdf._core.Object
# Here we change the module to discourage people from using that internal name
# Instead it will become pikepdf.objects.Object
Object.__module__ = __name__
ObjectType.__module__ = __name__


# type(Object) is the metaclass that pybind11 defines; we wish to extend that
# pylint cannot see the C++ metaclass definition and is thoroughly confused.
# pylint: disable=invalid-metaclass


class _ObjectMeta(type(Object)):  # type: ignore
    """Support instance checking."""

    def __instancecheck__(self, instance: Any) -> bool:
        # Note: since this class is a metaclass, self is a class object
        if type(instance) != Object:
            return False
        return self.object_type == instance._type_code


class _NameObjectMeta(_ObjectMeta):
    """Support usage pikepdf.Name.Whatever -> Name('/Whatever')."""

    def __getattr__(self, attr: str) -> Name:
        if attr.startswith('_') or attr == 'object_type':
            return getattr(_ObjectMeta, attr)
        return Name('/' + attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        # No need for a symmetric .startswith('_'). To prevent user error, we
        # simply don't allow mucking with the pikepdf.Name class's attributes.
        # There is no reason to ever assign to them.
        raise AttributeError(
            "Attributes may not be set on pikepdf.Name. Perhaps you meant to "
            "modify a Dictionary rather than a Name?"
        )

    def __getitem__(self, item: str) -> None:
        if item.startswith('/'):
            item = item[1:]
        raise TypeError(
            "pikepdf.Name is not subscriptable. You probably meant:\n"
            f"    pikepdf.Name.{item}\n"
            "or\n"
            f"    pikepdf.Name('/{item}')\n"
        )


class Name(Object, metaclass=_NameObjectMeta):
    """Construct a PDF Name object.

    Names can be constructed with two notations:

        1. ``Name.Resources``

        2. ``Name('/Resources')``

    The two are semantically equivalent. The former is preferred for names
    that are normally expected to be in a PDF. The latter is preferred for
    dynamic names and attributes.
    """

    object_type = ObjectType.name_

    def __new__(cls, name: str | Name) -> Name:
        """Construct a PDF Name."""
        # QPDF_Name::unparse ensures that names are always saved in a UTF-8
        # compatible way, so we only need to guard the input.
        if isinstance(name, bytes):
            raise TypeError("Name should be str")
        if isinstance(name, Name):
            return name  # Names are immutable so we can return a reference
        return _core._new_name(name)

    @classmethod
    def random(cls, len_: int = 16, prefix: str = '') -> Name:
        """Generate a cryptographically strong, random, valid PDF Name.

        If you are inserting a new name into a PDF (for example,
        name for a new image), you can use this function to generate a
        cryptographically strong random name that is almost certainly already
        not already in the PDF, and not colliding with other existing names.

        This function uses Python's secrets.token_urlsafe, which returns a
        URL-safe encoded random number of the desired length. An optional
        *prefix* may be prepended. (The encoding is ultimately done with
        :func:`base64.urlsafe_b64encode`.) Serendipitously, URL-safe is also
        PDF-safe.

        When the length parameter is 16 (16 random bytes or 128 bits), the result
        is probably globally unique and can be treated as never colliding with
        other names.

        The length of the returned string may vary because it is encoded,
        but will always have ``8 * len_`` random bits.

        Args:
            len_: The length of the random string.
            prefix: A prefix to prepend to the random string.
        """
        random_string = token_urlsafe(len_)
        return _core._new_name(f"/{prefix}{random_string}")


class Operator(Object, metaclass=_ObjectMeta):
    """Construct an operator for use in a content stream.

    An Operator is one of a limited set of commands that can appear in PDF content
    streams (roughly the mini-language that draws objects, lines and text on a
    virtual PDF canvas). The commands :func:`parse_content_stream` and
    :func:`unparse_content_stream` create and expect Operators respectively, along
    with their operands.

    pikepdf uses the special Operator "INLINE IMAGE" to denote an inline image
    in a content stream.
    """

    object_type = ObjectType.operator

    def __new__(cls, name: str) -> Operator:
        """Construct an operator."""
        return cast('Operator', _core._new_operator(name))


class String(Object, metaclass=_ObjectMeta):
    """Construct a PDF String object."""

    object_type = ObjectType.string

    def __new__(cls, s: str | bytes) -> String:
        """Construct a PDF String.

        Args:
            s: The string to use. String will be encoded for
                PDF, bytes will be constructed without encoding.
        """
        if isinstance(s, bytes):
            return _core._new_string(s)
        return _core._new_string_utf8(s)


class Array(Object, metaclass=_ObjectMeta):
    """Construct a PDF Array object."""

    object_type = ObjectType.array

    def __new__(cls, a: Iterable | Rectangle | Matrix | None = None) -> Array:
        """Construct a PDF Array.

        Args:
            a: An iterable of objects. All objects must be either
                `pikepdf.Object` or convertible to `pikepdf.Object`.
        """
        if isinstance(a, (str, bytes)):
            raise TypeError('Strings cannot be converted to arrays of chars')

        if a is None:
            a = []
        elif isinstance(a, (Rectangle, Matrix)):
            return a.as_array()
        elif isinstance(a, Array):
            return cast(Array, a.__copy__())
        return _core._new_array(a)


class Dictionary(Object, metaclass=_ObjectMeta):
    """Construct a PDF Dictionary object."""

    object_type = ObjectType.dictionary

    def __new__(cls, d: Mapping | None = None, **kwargs) -> Dictionary:
        """Construct a PDF Dictionary.

        Works from either a Python ``dict`` or keyword arguments.

        These two examples are equivalent:

        .. code-block:: python

            pikepdf.Dictionary({'/NameOne': 1, '/NameTwo': 'Two'})

            pikepdf.Dictionary(NameOne=1, NameTwo='Two')

        In either case, the keys must be strings, and the strings
        correspond to the desired Names in the PDF Dictionary. The values
        must all be convertible to `pikepdf.Object`.
        """
        if kwargs and d is not None:
            raise ValueError('Cannot use both a mapping object and keyword args')
        if kwargs:
            # Add leading slash
            # Allows Dictionary(MediaBox=(0,0,1,1), Type=Name('/Page')...
            return _core._new_dictionary({('/' + k): v for k, v in kwargs.items()})
        if isinstance(d, Dictionary):
            # Already a dictionary
            return d.__copy__()
        if not d:
            d = {}
        if d and any(key == '/' or not key.startswith('/') for key in d.keys()):
            raise KeyError("Dictionary created from strings must begin with '/'")
        return _core._new_dictionary(d)


class Stream(Object, metaclass=_ObjectMeta):
    """Construct a PDF Stream object."""

    object_type = ObjectType.stream

    def __new__(cls, owner: Pdf, data: bytes | None = None, d=None, **kwargs) -> Stream:
        """Create a new stream object.

        Streams stores arbitrary binary data and may or may not be compressed.
        It also may or may not be a page or Form XObject's content stream.

        A stream dictionary is like a pikepdf.Dictionary or Python dict, except
        it has a binary payload of data attached. The dictionary describes
        how the data is compressed or encoded.

        The dictionary may be initialized just like pikepdf.Dictionary is initialized,
        using a mapping object or keyword arguments.

        Args:
            owner: The Pdf to which this stream shall be attached.
            data: The data bytes for the stream.
            d: An optional mapping object that will be used to construct the stream's
                dictionary.
            kwargs: Keyword arguments that will define the stream dictionary. Do not set
                /Length here as pikepdf will manage this value. Set /Filter
                if the data is already encoded in some format.

        Examples:
            Using kwargs:
                >>> pdf = pikepdf.Pdf.new()
                >>> s1 = pikepdf.Stream(
                ...     pdf,
                ...     b"uncompressed image data",
                ...     BitsPerComponent=8,
                ...     ColorSpace=pikepdf.Name.DeviceRGB,
                ... )
            Using dict:
                >>> pdf = pikepdf.Pdf.new()
                >>> d = pikepdf.Dictionary(Key1=1, Key2=2)
                >>> s2 = pikepdf.Stream(
                ...     pdf,
                ...     b"data",
                ...     d
                ... )

        .. versionchanged:: 2.2
            Support creation of ``pikepdf.Stream`` from existing dictionary.

        .. versionchanged:: 3.0
            ``obj`` argument was removed; use ``data``.
        """
        if data is None:
            raise TypeError("Must make Stream from binary data")

        stream_dict = None
        if d or kwargs:
            stream_dict = Dictionary(d, **kwargs)

        stream = _core._new_stream(owner, data)
        if stream_dict:
            stream.stream_dict = stream_dict
        return stream
