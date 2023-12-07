# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""PDF metadata handling."""

from __future__ import annotations

import logging
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Set
from warnings import warn

from lxml import etree
from lxml.etree import QName, XMLSyntaxError

from .. import Name, Stream, String
from .. import __version__ as pikepdf_version
from .._xml import parse_xml

if sys.version_info < (3, 9):  # pragma: no cover
    from typing import Iterable, MutableMapping
else:
    from collections.abc import Iterable, MutableMapping

if TYPE_CHECKING:  # pragma: no cover
    from pikepdf import Pdf


XMP_NS_DC = "http://purl.org/dc/elements/1.1/"
XMP_NS_PDF = "http://ns.adobe.com/pdf/1.3/"
XMP_NS_PDFA_ID = "http://www.aiim.org/pdfa/ns/id/"
XMP_NS_PDFA_EXTENSION = "http://www.aiim.org/pdfa/ns/extension/"
XMP_NS_PDFA_PROPERTY = "http://www.aiim.org/pdfa/ns/property#"
XMP_NS_PDFA_SCHEMA = "http://www.aiim.org/pdfa/ns/schema#"
XMP_NS_PDFUA_ID = "http://www.aiim.org/pdfua/ns/id/"
XMP_NS_PDFX_ID = "http://www.npes.org/pdfx/ns/id/"
XMP_NS_PHOTOSHOP = "http://ns.adobe.com/photoshop/1.0/"
XMP_NS_PRISM = "http://prismstandard.org/namespaces/basic/1.0/"
XMP_NS_PRISM2 = "http://prismstandard.org/namespaces/basic/2.0/"
XMP_NS_PRISM3 = "http://prismstandard.org/namespaces/basic/3.0/"
XMP_NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS_XMP = "http://ns.adobe.com/xap/1.0/"
XMP_NS_XMP_MM = "http://ns.adobe.com/xap/1.0/mm/"
XMP_NS_XMP_RIGHTS = "http://ns.adobe.com/xap/1.0/rights/"

DEFAULT_NAMESPACES: list[tuple[str, str]] = [
    ('adobe:ns:meta/', 'x'),
    (XMP_NS_DC, 'dc'),
    (XMP_NS_PDF, 'pdf'),
    (XMP_NS_PDFA_ID, 'pdfaid'),
    (XMP_NS_PDFA_EXTENSION, 'pdfaExtension'),
    (XMP_NS_PDFA_PROPERTY, 'pdfaProperty'),
    (XMP_NS_PDFA_SCHEMA, 'pdfaSchema'),
    (XMP_NS_PDFUA_ID, 'pdfuaid'),
    (XMP_NS_PDFX_ID, 'pdfxid'),
    (XMP_NS_PHOTOSHOP, 'photoshop'),
    (XMP_NS_PRISM, 'prism'),
    (XMP_NS_PRISM2, 'prism2'),
    (XMP_NS_PRISM3, 'prism3'),
    (XMP_NS_RDF, 'rdf'),
    (XMP_NS_XMP, 'xmp'),
    (XMP_NS_XMP_MM, 'xmpMM'),
    (XMP_NS_XMP_RIGHTS, 'xmpRights'),
    ('http://crossref.org/crossmark/1.0/', 'crossmark'),
    ('http://www.niso.org/schemas/jav/1.0/', 'jav'),
    ('http://ns.adobe.com/pdfx/1.3/', 'pdfx'),
    ('http://www.niso.org/schemas/ali/1.0/', 'ali'),
]

for _uri, _prefix in DEFAULT_NAMESPACES:
    etree.register_namespace(_prefix, _uri)

# This one should not be registered
XMP_NS_XML = "http://www.w3.org/XML/1998/namespace"

XPACKET_BEGIN = b"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n"""

XMP_EMPTY = b"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="pikepdf">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
 </rdf:RDF>
</x:xmpmeta>
"""

XPACKET_END = b"""\n<?xpacket end="w"?>\n"""


class XmpContainer(NamedTuple):
    """Map XMP container object to suitable Python container."""

    rdf_type: str
    py_type: type
    insert_fn: Callable[..., None]


log = logging.getLogger(__name__)


class NeverRaise(Exception):
    """An exception that is never raised."""


class AltList(list):
    """XMP AltList container."""


XMP_CONTAINERS = [
    XmpContainer('Alt', AltList, AltList.append),
    XmpContainer('Bag', set, set.add),
    XmpContainer('Seq', list, list.append),
]

LANG_ALTS = frozenset(
    [
        str(QName(XMP_NS_DC, 'title')),
        str(QName(XMP_NS_DC, 'description')),
        str(QName(XMP_NS_DC, 'rights')),
        str(QName(XMP_NS_XMP_RIGHTS, 'UsageTerms')),
    ]
)

# These are the illegal characters in XML 1.0. (XML 1.1 is a bit more permissive,
# but we'll be strict to ensure wider compatibility.)
re_xml_illegal_chars = re.compile(
    r"(?u)[^\x09\x0A\x0D\x20-\U0000D7FF\U0000E000-\U0000FFFD\U00010000-\U0010FFFF]"
)
re_xml_illegal_bytes = re.compile(br"[^\x09\x0A\x0D\x20-\xFF]|&#0;")

# Might want to check re_xml_illegal_bytes for patterns such as:
# br"&#(?:[0-9]|0[0-9]|1[0-9]|2[0-9]|3[0-1]
#   |x[0-9A-Fa-f]|x0[0-9A-Fa-f]|x1[0-9A-Fa-f]);"


def _parser_basic(xml: bytes):
    return parse_xml(BytesIO(xml))


def _parser_strip_illegal_bytes(xml: bytes):
    return parse_xml(BytesIO(re_xml_illegal_bytes.sub(b'', xml)))


def _parser_recovery(xml: bytes):
    return parse_xml(BytesIO(xml), recover=True)


def _parser_replace_with_empty_xmp(_xml: bytes = b''):
    log.warning("Error occurred parsing XMP, replacing with empty XMP.")
    return _parser_basic(XMP_EMPTY)


def _clean(s: str | Iterable[str], joiner: str = '; ') -> str:
    """Ensure an object can safely be inserted in a XML tag body.

    If we still have a non-str object at this point, the best option is to
    join it, because it's apparently calling for a new node in a place that
    isn't allowed in the spec or not supported.
    """
    if not isinstance(s, str):
        if isinstance(s, Iterable):
            warn(f"Merging elements of {s}")
            if isinstance(s, Set):
                s = joiner.join(sorted(s))
            else:
                s = joiner.join(s)
        else:
            raise TypeError("object must be a string or iterable of strings")
    return re_xml_illegal_chars.sub('', s)


def encode_pdf_date(d: datetime) -> str:
    """Encode Python datetime object as PDF date string.

    From Adobe pdfmark manual:
    (D:YYYYMMDDHHmmSSOHH'mm')
    D: is an optional prefix. YYYY is the year. All fields after the year are
    optional. MM is the month (01-12), DD is the day (01-31), HH is the
    hour (00-23), mm are the minutes (00-59), and SS are the seconds
    (00-59). The remainder of the string defines the relation of local
    time to GMT. O is either + for a positive difference (local time is
    later than GMT) or - (minus) for a negative difference. HH' is the
    absolute value of the offset from GMT in hours, and mm' is the
    absolute value of the offset in minutes. If no GMT information is
    specified, the relation between the specified time and GMT is
    considered unknown. Regardless of whether or not GMT
    information is specified, the remainder of the string should specify
    the local time.

    'D:' is required in PDF/A, so we always add it.
    """
    # The formatting of %Y is not consistent as described in
    # https://bugs.python.org/issue13305 and underspecification in libc.
    # So explicitly format the year with leading zeros
    s = f"D:{d.year:04d}"
    s += d.strftime(r'%m%d%H%M%S')
    tz = d.strftime('%z')
    if tz:
        sign, tz_hours, tz_mins = tz[0], tz[1:3], tz[3:5]
        s += f"{sign}{tz_hours}'{tz_mins}'"
    return s


def decode_pdf_date(s: str) -> datetime:
    """Decode a pdfmark date to a Python datetime object.

    A pdfmark date is a string in a paritcular format. See the pdfmark
    Reference for the specification.
    """
    if isinstance(s, String):
        s = str(s)
    if s.startswith('D:'):
        s = s[2:]

    # Literal Z00'00', is incorrect but found in the wild,
    # probably made by OS X Quartz -- standardize
    if s.endswith("Z00'00'"):
        s = s.replace("Z00'00'", '+0000')
    elif s.endswith('Z'):
        s = s.replace('Z', '+0000')
    s = s.replace("'", "")  # Remove apos from PDF time strings
    try:
        return datetime.strptime(s, r'%Y%m%d%H%M%S%z')
    except ValueError:
        return datetime.strptime(s, r'%Y%m%d%H%M%S')


class Converter(ABC):
    """XMP <-> DocumentInfo converter."""

    @staticmethod
    @abstractmethod
    def xmp_from_docinfo(docinfo_val: str | None) -> Any:  # type: ignore
        """Derive XMP metadata from a DocumentInfo string."""

    @staticmethod
    @abstractmethod
    def docinfo_from_xmp(xmp_val: Any) -> str | None:
        """Derive a DocumentInfo value from equivalent XMP metadata."""


class AuthorConverter(Converter):
    """Convert XMP document authors to DocumentInfo."""

    @staticmethod
    def xmp_from_docinfo(docinfo_val: str | None) -> Any:  # type: ignore
        """Derive XMP authors info from DocumentInfo."""
        return [docinfo_val]

    @staticmethod
    def docinfo_from_xmp(xmp_val):
        """Derive DocumentInfo authors from XMP.

        XMP supports multiple author values, while DocumentInfo has a string,
        so we return the values separated by semi-colons.
        """
        if isinstance(xmp_val, str):
            return xmp_val
        if xmp_val is None or xmp_val == [None]:
            return None
        return '; '.join(author for author in xmp_val if author is not None)


class DateConverter(Converter):
    """Convert XMP dates to DocumentInfo."""

    @staticmethod
    def xmp_from_docinfo(docinfo_val):
        """Derive XMP date from DocumentInfo."""
        if docinfo_val == '':
            return ''
        return decode_pdf_date(docinfo_val).isoformat()

    @staticmethod
    def docinfo_from_xmp(xmp_val):
        """Derive DocumentInfo from XMP."""
        if xmp_val.endswith('Z'):
            xmp_val = xmp_val[:-1] + '+00:00'
        try:
            dateobj = datetime.fromisoformat(xmp_val)
        except IndexError:
            # PyPy 3.8 may raise IndexError - convert to ValueError
            raise ValueError(f"Invalid isoformat string: '{xmp_val}'") from None
        return encode_pdf_date(dateobj)


class DocinfoMapping(NamedTuple):
    """Map DocumentInfo keys to their XMP equivalents, along with converter."""

    ns: str
    key: str
    name: Name
    converter: type[Converter] | None


def ensure_loaded(fn):
    """Ensure the XMP has been loaded and parsed.

    TODO: Can this be removed? Why allow the uninit'ed state to even exist?
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._xmp:
            self._load()
        return fn(self, *args, **kwargs)

    return wrapper


class PdfMetadata(MutableMapping):
    """Read and edit the metadata associated with a PDF.

    The PDF specification contain two types of metadata, the newer XMP
    (Extensible Metadata Platform, XML-based) and older DocumentInformation
    dictionary. The PDF 2.0 specification removes the DocumentInformation
    dictionary.

    This primarily works with XMP metadata, but includes methods to generate
    XMP from DocumentInformation and will also coordinate updates to
    DocumentInformation so that the two are kept consistent.

    XMP metadata fields may be accessed using the full XML namespace URI or
    the short name. For example ``metadata['dc:description']``
    and ``metadata['{http://purl.org/dc/elements/1.1/}description']``
    both refer to the same field. Several common XML namespaces are registered
    automatically.

    See the XMP specification for details of allowable fields.

    To update metadata, use a with block.

    Example:
        >>> with pdf.open_metadata() as records:
                records['dc:title'] = 'New Title'

    See Also:
        :meth:`pikepdf.Pdf.open_metadata`
    """

    DOCINFO_MAPPING: list[DocinfoMapping] = [
        DocinfoMapping(XMP_NS_DC, 'creator', Name.Author, AuthorConverter),
        DocinfoMapping(XMP_NS_DC, 'description', Name.Subject, None),
        DocinfoMapping(XMP_NS_DC, 'title', Name.Title, None),
        DocinfoMapping(XMP_NS_PDF, 'Keywords', Name.Keywords, None),
        DocinfoMapping(XMP_NS_PDF, 'Producer', Name.Producer, None),
        DocinfoMapping(XMP_NS_XMP, 'CreateDate', Name.CreationDate, DateConverter),
        DocinfoMapping(XMP_NS_XMP, 'CreatorTool', Name.Creator, None),
        DocinfoMapping(XMP_NS_XMP, 'ModifyDate', Name.ModDate, DateConverter),
    ]

    NS: dict[str, str] = {prefix: uri for uri, prefix in DEFAULT_NAMESPACES}
    REVERSE_NS: dict[str, str] = dict(DEFAULT_NAMESPACES)

    _PARSERS_OVERWRITE_INVALID_XML: Iterable[Callable[[bytes], Any]] = [
        _parser_basic,
        _parser_strip_illegal_bytes,
        _parser_recovery,
        _parser_replace_with_empty_xmp,
    ]
    _PARSERS_STANDARD: Iterable[Callable[[bytes], Any]] = [_parser_basic]

    @classmethod
    def register_xml_namespace(cls, uri, prefix):
        """Register a new XML/XMP namespace.

        Arguments:
            uri: The long form of the namespace.
            prefix: The alias to use when interpreting XMP.
        """
        cls.NS[prefix] = uri
        cls.REVERSE_NS[uri] = prefix
        etree.register_namespace(_prefix, _uri)

    def __init__(
        self,
        pdf: Pdf,
        pikepdf_mark: bool = True,
        sync_docinfo: bool = True,
        overwrite_invalid_xml: bool = True,
    ):
        """Construct PdfMetadata. Use Pdf.open_metadata() instead."""
        self._pdf = pdf
        self._xmp = None
        self.mark = pikepdf_mark
        self.sync_docinfo = sync_docinfo
        self._updating = False
        self.overwrite_invalid_xml = overwrite_invalid_xml

    def load_from_docinfo(
        self, docinfo, delete_missing: bool = False, raise_failure: bool = False
    ) -> None:
        """Populate the XMP metadata object with DocumentInfo.

        Arguments:
            docinfo: a DocumentInfo, e.g pdf.docinfo
            delete_missing: if the entry is not DocumentInfo, delete the equivalent
                from XMP
            raise_failure: if True, raise any failure to convert docinfo;
                otherwise warn and continue

        A few entries in the deprecated DocumentInfo dictionary are considered
        approximately equivalent to certain XMP records. This method copies
        those entries into the XMP metadata.
        """

        def warn_or_raise(msg, e=None):
            if raise_failure:
                raise ValueError(msg) from e
            warn(msg)

        for uri, shortkey, docinfo_name, converter in self.DOCINFO_MAPPING:
            qname = QName(uri, shortkey)
            # docinfo might be a dict or pikepdf.Dictionary, so lookup keys
            # by str(Name)
            val = docinfo.get(str(docinfo_name))
            if val is None:
                if delete_missing and qname in self:
                    del self[qname]
                continue
            try:
                val = str(val)
                if converter:
                    val = converter.xmp_from_docinfo(val)
                if not val:
                    continue
                self._setitem(qname, val, True)
            except (ValueError, AttributeError, NotImplementedError) as e:
                warn_or_raise(
                    f"The metadata field {docinfo_name} could not be copied to XMP", e
                )
        valid_docinfo_names = {
            str(docinfo_name) for _, _, docinfo_name, _ in self.DOCINFO_MAPPING
        }
        extra_docinfo_names = {str(k) for k in docinfo.keys()} - valid_docinfo_names
        for extra in extra_docinfo_names:
            warn_or_raise(
                f"The metadata field {extra} with value '{repr(docinfo.get(extra))}' "
                "has no XMP equivalent, so it was discarded",
            )

    def _load(self) -> None:
        try:
            data = self._pdf.Root.Metadata.read_bytes()
        except AttributeError:
            data = b''
        self._load_from(data)

    def _load_from(self, data: bytes) -> None:
        if data.strip() == b'':
            data = XMP_EMPTY  # on some platforms lxml chokes on empty documents

        parsers = (
            self._PARSERS_OVERWRITE_INVALID_XML
            if self.overwrite_invalid_xml
            else self._PARSERS_STANDARD
        )

        for parser in parsers:
            try:
                self._xmp = parser(data)
            except (
                XMLSyntaxError
                if self.overwrite_invalid_xml
                else NeverRaise  # type: ignore
            ) as e:
                if str(e).startswith("Start tag expected, '<' not found") or str(
                    e
                ).startswith("Document is empty"):
                    self._xmp = _parser_replace_with_empty_xmp()
                    break
            else:
                break

        if self._xmp is not None:
            try:
                pis = self._xmp.xpath('/processing-instruction()')
                for pi in pis:
                    etree.strip_tags(self._xmp, pi.tag)
                self._get_rdf_root()
            except (
                Exception  # pylint: disable=broad-except
                if self.overwrite_invalid_xml
                else NeverRaise
            ) as e:
                log.warning("Error occurred parsing XMP", exc_info=e)
                self._xmp = _parser_replace_with_empty_xmp()
        else:
            log.warning("Error occurred parsing XMP")
            self._xmp = _parser_replace_with_empty_xmp()

    @ensure_loaded
    def __enter__(self):
        """Open metadata for editing."""
        self._updating = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close metadata and apply changes."""
        try:
            if exc_type is not None:
                return
            self._apply_changes()
        finally:
            self._updating = False

    def _update_docinfo(self):
        """Update the PDF's DocumentInfo dictionary to match XMP metadata.

        The standard mapping is described here:
            https://www.pdfa.org/pdfa-metadata-xmp-rdf-dublin-core/
        """
        # Touch object to ensure it exists
        self._pdf.docinfo  # pylint: disable=pointless-statement
        for uri, element, docinfo_name, converter in self.DOCINFO_MAPPING:
            qname = QName(uri, element)
            try:
                value = self[qname]
            except KeyError:
                if docinfo_name in self._pdf.docinfo:
                    del self._pdf.docinfo[docinfo_name]
                continue
            if converter:
                try:
                    value = converter.docinfo_from_xmp(value)
                except ValueError:
                    warn(
                        f"The DocumentInfo field {docinfo_name} could not be "
                        "updated from XMP"
                    )
                    value = None
                except Exception as e:
                    raise ValueError(
                        "An error occurred while updating DocumentInfo field "
                        f"{docinfo_name} from XMP {qname} with value {value}"
                    ) from e
            if value is None:
                if docinfo_name in self._pdf.docinfo:
                    del self._pdf.docinfo[docinfo_name]
                continue
            value = _clean(value)
            try:
                # Try to save pure ASCII
                self._pdf.docinfo[docinfo_name] = value.encode('ascii')
            except UnicodeEncodeError:
                # qpdf will serialize this as a UTF-16 with BOM string
                self._pdf.docinfo[docinfo_name] = value

    def _get_xml_bytes(self, xpacket=True):
        data = BytesIO()
        if xpacket:
            data.write(XPACKET_BEGIN)
        self._xmp.write(data, encoding='utf-8', pretty_print=True)
        if xpacket:
            data.write(XPACKET_END)
        data.seek(0)
        xml_bytes = data.read()
        return xml_bytes

    def _apply_changes(self):
        """Serialize our changes back to the PDF in memory.

        Depending how we are initialized, leave our metadata mark and producer.
        """
        if self.mark:
            # We were asked to mark the file as being edited by pikepdf
            self._setitem(
                QName(XMP_NS_XMP, 'MetadataDate'),
                datetime.now(timezone.utc).isoformat(),
                applying_mark=True,
            )
            self._setitem(
                QName(XMP_NS_PDF, 'Producer'),
                'pikepdf ' + pikepdf_version,
                applying_mark=True,
            )
        xml = self._get_xml_bytes()
        self._pdf.Root.Metadata = Stream(self._pdf, xml)
        self._pdf.Root.Metadata[Name.Type] = Name.Metadata
        self._pdf.Root.Metadata[Name.Subtype] = Name.XML
        if self.sync_docinfo:
            self._update_docinfo()

    @classmethod
    def _qname(cls, name: QName | str) -> str:
        """Convert name to an XML QName.

        e.g. pdf:Producer -> {http://ns.adobe.com/pdf/1.3/}Producer
        """
        if isinstance(name, QName):
            return str(name)
        if not isinstance(name, str):
            raise TypeError(f"{name} must be str")
        if name == '':
            return name
        if name.startswith('{'):
            return name
        try:
            prefix, tag = name.split(':', maxsplit=1)
        except ValueError:
            # If missing the namespace, put it in the top level namespace
            # To do this completely correct we actually need to figure out
            # the namespace based on context defined by parent tags. That
            #   https://www.w3.org/2001/tag/doc/qnameids.html
            prefix, tag = 'x', name
        uri = cls.NS[prefix]
        return str(QName(uri, tag))

    def _prefix_from_uri(self, uriname):
        """Given a fully qualified XML name, find a prefix.

        e.g. {http://ns.adobe.com/pdf/1.3/}Producer -> pdf:Producer
        """
        uripart, tag = uriname.split('}', maxsplit=1)
        uri = uripart.replace('{', '')
        return self.REVERSE_NS[uri] + ':' + tag

    def _get_subelements(self, node):
        """Gather the sub-elements attached to a node.

        Gather rdf:Bag and and rdf:Seq into set and list respectively. For
        alternate languages values, take the first language only for
        simplicity.
        """
        items = node.find('rdf:Alt', self.NS)
        if items is not None:
            try:
                return items[0].text
            except IndexError:
                return ''

        for xmlcontainer, container, insertfn in XMP_CONTAINERS:
            items = node.find(f'rdf:{xmlcontainer}', self.NS)
            if items is None:
                continue
            result = container()
            for item in items:
                insertfn(result, item.text)
            return result
        return ''

    def _get_rdf_root(self):
        rdf = self._xmp.find('.//rdf:RDF', self.NS)
        if rdf is None:
            rdf = self._xmp.getroot()
            if not rdf.tag == '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF':
                raise ValueError("Metadata seems to be XML but not XMP")
        return rdf

    def _get_elements(self, name: str | QName = ''):
        """Get elements from XMP.

        Core routine to find elements matching name within the XMP and yield
        them.

        For XMP spec 7.9.2.2, rdf:Description with property attributes,
        we yield the node which will have the desired as one of its attributes.
        qname is returned so that the node.attrib can be used to locate the
        source.

        For XMP spec 7.5, simple valued XMP properties, we yield the node,
        None, and the value. For structure or array valued properties we gather
        the elements. We ignore qualifiers.

        Args:
            name: a prefixed name or QName to look for within the
                data section of the XMP; looks for all data keys if omitted

        Yields:
            tuple: (node, qname_attrib, value, parent_node)

        """
        qname = self._qname(name)
        rdf = self._get_rdf_root()
        for rdfdesc in rdf.findall('rdf:Description[@rdf:about=""]', self.NS):
            if qname and qname in rdfdesc.keys():
                yield (rdfdesc, qname, rdfdesc.get(qname), rdf)
            elif not qname:
                for k, v in rdfdesc.items():
                    if v:
                        yield (rdfdesc, k, v, rdf)
            xpath = qname if name else '*'
            for node in rdfdesc.findall(xpath, self.NS):
                if node.text and node.text.strip():
                    yield (node, None, node.text, rdfdesc)
                    continue
                values = self._get_subelements(node)
                yield (node, None, values, rdfdesc)

    def _get_element_values(self, name=''):
        yield from (v[2] for v in self._get_elements(name))

    @ensure_loaded
    def __contains__(self, key: str | QName):
        """Test if XMP key is in metadata."""
        return any(self._get_element_values(key))

    @ensure_loaded
    def __getitem__(self, key: str | QName):
        """Retrieve XMP metadata for key."""
        try:
            return next(self._get_element_values(key))
        except StopIteration:
            raise KeyError(key) from None

    @ensure_loaded
    def __iter__(self):
        """Iterate through XMP metadata attributes and nodes."""
        for node, attrib, _val, _parents in self._get_elements():
            if attrib:
                yield attrib
            else:
                yield node.tag

    @ensure_loaded
    def __len__(self):
        """Return number of items in metadata."""
        return len(list(iter(self)))

    def _setitem(
        self,
        key: str | QName,
        val: set[str] | list[str] | str,
        applying_mark: bool = False,
    ):
        if not self._updating:
            raise RuntimeError("Metadata not opened for editing, use with block")

        qkey = self._qname(key)
        self._setitem_check_args(key, val, applying_mark, qkey)

        try:
            # Update existing node
            self._setitem_update(key, val, qkey)
        except StopIteration:
            # Insert a new node
            self._setitem_insert(key, val)

    def _setitem_check_args(self, key, val, applying_mark: bool, qkey: str) -> None:
        if (
            self.mark
            and not applying_mark
            and qkey
            in (
                self._qname('xmp:MetadataDate'),
                self._qname('pdf:Producer'),
            )
        ):
            # Complain if user writes self[pdf:Producer] = ... and because it will
            # be overwritten on save, unless self._updating_mark, in which case
            # the action was initiated internally
            log.warning(
                f"Update to {key} will be overwritten because metadata was opened "
                "with set_pikepdf_as_editor=True"
            )
        if isinstance(val, str) and qkey in (self._qname('dc:creator')):
            log.error(f"{key} should be set to a list of strings")

    def _setitem_add_array(self, node, items: Iterable) -> None:
        rdf_type = next(
            c.rdf_type for c in XMP_CONTAINERS if isinstance(items, c.py_type)
        )
        seq = etree.SubElement(node, str(QName(XMP_NS_RDF, rdf_type)))
        tag_attrib: dict[str, str] | None = None
        if rdf_type == 'Alt':
            tag_attrib = {str(QName(XMP_NS_XML, 'lang')): 'x-default'}
        for item in items:
            el = etree.SubElement(seq, str(QName(XMP_NS_RDF, 'li')), attrib=tag_attrib)
            el.text = _clean(item)

    def _setitem_update(self, key, val, qkey):
        # Locate existing node to replace
        node, attrib, _oldval, _parent = next(self._get_elements(key))
        if attrib:
            if not isinstance(val, str):
                if qkey == self._qname('dc:creator'):
                    # dc:creator incorrectly created as an attribute - we're
                    # replacing it anyway, so remove the old one
                    del node.attrib[qkey]
                    self._setitem_add_array(node, _clean(val))
                else:
                    raise TypeError(f"Setting {key} to {val} with type {type(val)}")
            else:
                node.set(attrib, _clean(val))
        elif isinstance(val, (list, set)):
            for child in node.findall('*'):
                node.remove(child)
            self._setitem_add_array(node, val)
        elif isinstance(val, str):
            for child in node.findall('*'):
                node.remove(child)
            if str(self._qname(key)) in LANG_ALTS:
                self._setitem_add_array(node, AltList([_clean(val)]))
            else:
                node.text = _clean(val)
        else:
            raise TypeError(f"Setting {key} to {val} with type {type(val)}")

    def _setitem_insert(self, key, val):
        rdf = self._get_rdf_root()
        if str(self._qname(key)) in LANG_ALTS:
            val = AltList([_clean(val)])
        if isinstance(val, (list, set)):
            rdfdesc = etree.SubElement(
                rdf,
                str(QName(XMP_NS_RDF, 'Description')),
                attrib={str(QName(XMP_NS_RDF, 'about')): ''},
            )
            node = etree.SubElement(rdfdesc, self._qname(key))
            self._setitem_add_array(node, val)
        elif isinstance(val, str):
            _rdfdesc = etree.SubElement(
                rdf,
                str(QName(XMP_NS_RDF, 'Description')),
                attrib={
                    QName(XMP_NS_RDF, 'about'): '',
                    self._qname(key): _clean(val),
                },
            )
        else:
            raise TypeError(f"Setting {key} to {val} with type {type(val)}") from None

    @ensure_loaded
    def __setitem__(self, key: str | QName, val: set[str] | list[str] | str):
        """Set XMP metadata key to value."""
        return self._setitem(key, val, False)

    @ensure_loaded
    def __delitem__(self, key: str | QName):
        """Delete item from XMP metadata."""
        if not self._updating:
            raise RuntimeError("Metadata not opened for editing, use with block")
        try:
            node, attrib, _oldval, parent = next(self._get_elements(key))
            if attrib:  # Inline
                del node.attrib[attrib]
                if (
                    len(node.attrib) == 1
                    and len(node) == 0
                    and QName(XMP_NS_RDF, 'about') in node.attrib
                ):
                    # The only thing left on this node is rdf:about="", so remove it
                    parent.remove(node)
            else:
                parent.remove(node)
        except StopIteration:
            raise KeyError(key) from None

    @property
    def pdfa_status(self) -> str:
        """Return the PDF/A conformance level claimed by this PDF, or False.

        A PDF may claim to PDF/A compliant without this being true. Use an
        independent verifier such as veraPDF to test if a PDF is truly
        conformant.

        Returns:
            The conformance level of the PDF/A, or an empty string if the
            PDF does not claim PDF/A conformance. Possible valid values
            are: 1A, 1B, 2A, 2B, 2U, 3A, 3B, 3U.
        """
        # do same as @ensure_loaded - mypy can't handle decorated property
        if not self._xmp:
            self._load()

        key_part = QName(XMP_NS_PDFA_ID, 'part')
        key_conformance = QName(XMP_NS_PDFA_ID, 'conformance')
        try:
            return self[key_part] + self[key_conformance]
        except KeyError:
            return ''

    @property
    def pdfx_status(self) -> str:
        """Return the PDF/X conformance level claimed by this PDF, or False.

        A PDF may claim to PDF/X compliant without this being true. Use an
        independent verifier such as veraPDF to test if a PDF is truly
        conformant.

        Returns:
            The conformance level of the PDF/X, or an empty string if the
            PDF does not claim PDF/X conformance.
        """
        # do same as @ensure_loaded - mypy can't handle decorated property
        if not self._xmp:
            self._load()

        pdfx_version = QName(XMP_NS_PDFX_ID, 'GTS_PDFXVersion')
        try:
            return self[pdfx_version]
        except KeyError:
            return ''

    @ensure_loaded
    def __str__(self):
        """Convert XMP metadata to XML string."""
        return self._get_xml_bytes(xpacket=False).decode('utf-8')
