# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import IO, Any, AnyStr

from lxml.etree import XMLParser as _UnsafeXMLParser
from lxml.etree import _ElementTree
from lxml.etree import parse as _parse


class _XMLParser(_UnsafeXMLParser):
    def __init__(self, *args: Any, **kwargs: Any):
        # Prevent XXE attacks
        # https://rules.sonarsource.com/python/type/Vulnerability/RSPEC-2755
        kwargs['resolve_entities'] = False
        kwargs['no_network'] = True
        super().__init__(*args, **kwargs)


def parse_xml(source: AnyStr | IO[Any], recover: bool = False) -> _ElementTree:
    """Wrap lxml's parse to provide protection against XXE attacks."""
    parser = _XMLParser(recover=recover, remove_pis=False)
    return _parse(source, parser=parser)


__all__ = ['parse_xml']
