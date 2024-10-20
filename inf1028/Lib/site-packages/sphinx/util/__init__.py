"""Utility functions for Sphinx."""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
from importlib import import_module
from os import path
from typing import IO, Any
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

from sphinx.errors import ExtensionError, FiletypeNotFoundError
from sphinx.locale import __
from sphinx.util import index_entries as _index_entries
from sphinx.util import logging
from sphinx.util.console import strip_colors  # NoQA: F401
from sphinx.util.matching import patfilter  # NoQA: F401
from sphinx.util.nodes import (  # NoQA: F401
    caption_ref_re,
    explicit_title_re,
    nested_parse_with_titles,
    split_explicit_title,
)

# import other utilities; partly for backwards compatibility, so don't
# prune unused ones indiscriminately
from sphinx.util.osutil import (  # NoQA: F401
    SEP,
    copyfile,
    ensuredir,
    make_filename,
    os_path,
    relative_uri,
)

logger = logging.getLogger(__name__)

# Generally useful regular expressions.
ws_re: re.Pattern[str] = re.compile(r'\s+')
url_re: re.Pattern[str] = re.compile(r'(?P<schema>.+)://.*')


# High-level utility functions.

def docname_join(basedocname: str, docname: str) -> str:
    return posixpath.normpath(posixpath.join('/' + basedocname, '..', docname))[1:]


def get_filetype(source_suffix: dict[str, str], filename: str | os.PathLike) -> str:
    for suffix, filetype in source_suffix.items():
        if os.fspath(filename).endswith(suffix):
            # If default filetype (None), considered as restructuredtext.
            return filetype or 'restructuredtext'
    raise FiletypeNotFoundError


class FilenameUniqDict(dict):
    """
    A dictionary that automatically generates unique names for its keys,
    interpreted as filenames, and keeps track of a set of docnames they
    appear in.  Used for images and downloadable files in the environment.
    """

    def __init__(self) -> None:
        self._existing: set[str] = set()

    def add_file(self, docname: str, newfile: str) -> str:
        if newfile in self:
            self[newfile][0].add(docname)
            return self[newfile][1]
        uniquename = path.basename(newfile)
        base, ext = path.splitext(uniquename)
        i = 0
        while uniquename in self._existing:
            i += 1
            uniquename = f'{base}{i}{ext}'
        self[newfile] = ({docname}, uniquename)
        self._existing.add(uniquename)
        return uniquename

    def purge_doc(self, docname: str) -> None:
        for filename, (docs, unique) in list(self.items()):
            docs.discard(docname)
            if not docs:
                del self[filename]
                self._existing.discard(unique)

    def merge_other(self, docnames: set[str], other: dict[str, tuple[set[str], Any]]) -> None:
        for filename, (docs, _unique) in other.items():
            for doc in docs & set(docnames):
                self.add_file(doc, filename)

    def __getstate__(self) -> set[str]:
        return self._existing

    def __setstate__(self, state: set[str]) -> None:
        self._existing = state


def _md5(data: bytes = b'', **_kw: Any) -> hashlib._Hash:
    """Deprecated wrapper around hashlib.md5

    To be removed in Sphinx 9.0
    """
    return hashlib.md5(data, usedforsecurity=False)


def _sha1(data: bytes = b'', **_kw: Any) -> hashlib._Hash:
    """Deprecated wrapper around hashlib.sha1

    To be removed in Sphinx 9.0
    """
    return hashlib.sha1(data, usedforsecurity=False)


class DownloadFiles(dict):
    """A special dictionary for download files.

    .. important:: This class would be refactored in nearly future.
                   Hence don't hack this directly.
    """

    def add_file(self, docname: str, filename: str) -> str:
        if filename not in self:
            digest = hashlib.md5(filename.encode(), usedforsecurity=False).hexdigest()
            dest = f'{digest}/{os.path.basename(filename)}'
            self[filename] = (set(), dest)

        self[filename][0].add(docname)
        return self[filename][1]

    def purge_doc(self, docname: str) -> None:
        for filename, (docs, _dest) in list(self.items()):
            docs.discard(docname)
            if not docs:
                del self[filename]

    def merge_other(self, docnames: set[str], other: dict[str, tuple[set[str], Any]]) -> None:
        for filename, (docs, _dest) in other.items():
            for docname in docs & set(docnames):
                self.add_file(docname, filename)


# a regex to recognize coding cookies
_coding_re = re.compile(r'coding[:=]\s*([-\w.]+)')


class UnicodeDecodeErrorHandler:
    """Custom error handler for open() that warns and replaces."""

    def __init__(self, docname: str) -> None:
        self.docname = docname

    def __call__(self, error: UnicodeDecodeError) -> tuple[str, int]:
        linestart = error.object.rfind(b'\n', 0, error.start)
        lineend = error.object.find(b'\n', error.start)
        if lineend == -1:
            lineend = len(error.object)
        lineno = error.object.count(b'\n', 0, error.start) + 1
        logger.warning(__('undecodable source characters, replacing with "?": %r'),
                       (error.object[linestart + 1:error.start] + b'>>>' +
                        error.object[error.start:error.end] + b'<<<' +
                        error.object[error.end:lineend]),
                       location=(self.docname, lineno))
        return ('?', error.end)


# Low-level utility functions and classes.

class Tee:
    """
    File-like object writing to two streams.
    """

    def __init__(self, stream1: IO, stream2: IO) -> None:
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, text: str) -> None:
        self.stream1.write(text)
        self.stream2.write(text)

    def flush(self) -> None:
        if hasattr(self.stream1, 'flush'):
            self.stream1.flush()
        if hasattr(self.stream2, 'flush'):
            self.stream2.flush()


def parselinenos(spec: str, total: int) -> list[int]:
    """Parse a line number spec (such as "1,2,4-6") and return a list of
    wanted line numbers.
    """
    items = []
    parts = spec.split(',')
    for part in parts:
        try:
            begend = part.strip().split('-')
            if begend == ['', '']:
                raise ValueError
            if len(begend) == 1:
                items.append(int(begend[0]) - 1)
            elif len(begend) == 2:
                start = int(begend[0] or 1)  # left half open (cf. -10)
                end = int(begend[1] or max(start, total))  # right half open (cf. 10-)
                if start > end:  # invalid range (cf. 10-1)
                    raise ValueError
                items.extend(range(start - 1, end))
            else:
                raise ValueError
        except ValueError as exc:
            msg = f'invalid line number spec: {spec!r}'
            raise ValueError(msg) from exc

    return items


def import_object(objname: str, source: str | None = None) -> Any:
    """Import python object by qualname."""
    try:
        objpath = objname.split('.')
        modname = objpath.pop(0)
        obj = import_module(modname)
        for name in objpath:
            modname += '.' + name
            try:
                obj = getattr(obj, name)
            except AttributeError:
                obj = import_module(modname)

        return obj
    except (AttributeError, ImportError) as exc:
        if source:
            raise ExtensionError('Could not import %s (needed for %s)' %
                                 (objname, source), exc) from exc
        raise ExtensionError('Could not import %s' % objname, exc) from exc


def encode_uri(uri: str) -> str:
    split = list(urlsplit(uri))
    split[1] = split[1].encode('idna').decode('ascii')
    split[2] = quote_plus(split[2].encode(), '/')
    query = [(q, v.encode()) for (q, v) in parse_qsl(split[3])]
    split[3] = urlencode(query)
    return urlunsplit(split)


def isurl(url: str) -> bool:
    """Check *url* is URL or not."""
    return bool(url) and '://' in url


# deprecated name -> (object to return, canonical path or empty string)
_DEPRECATED_OBJECTS: dict[str, tuple[Any, str, tuple[int, int]]] = {
    'split_index_msg': (_index_entries.split_index_msg,
                        'sphinx.util.index_entries.split_index_msg',
                        (9, 0)),
    'split_into': (_index_entries.split_index_msg,
                   'sphinx.util.index_entries.split_into',
                   (9, 0)),
    'md5': (_md5, '', (9, 0)),
    'sha1': (_sha1, '', (9, 0)),
}


def __getattr__(name: str) -> Any:
    if name not in _DEPRECATED_OBJECTS:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)

    from sphinx.deprecation import _deprecation_warning

    deprecated_object, canonical_name, remove = _DEPRECATED_OBJECTS[name]
    _deprecation_warning(__name__, name, canonical_name, remove=remove)
    return deprecated_object
