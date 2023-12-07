# SPDX-FileCopyrightText: 2022 James R. Barlow, 2020 Matthias Erll

# SPDX-License-Identifier: MPL-2.0

"""Support for document outlines (e.g. table of contents)."""

from __future__ import annotations

from enum import Enum
from itertools import chain
from typing import Iterable, List, cast

from pikepdf import Array, Dictionary, Name, Object, Page, Pdf, String


class PageLocation(Enum):
    """Page view location definitions, from PDF spec."""

    XYZ = 1
    Fit = 2
    FitH = 3
    FitV = 4
    FitR = 5
    FitB = 6
    FitBH = 7
    FitBV = 8


PAGE_LOCATION_ARGS = {
    PageLocation.XYZ: ('left', 'top', 'zoom'),
    PageLocation.FitH: ('top',),
    PageLocation.FitV: ('left',),
    PageLocation.FitR: ('left', 'bottom', 'right', 'top'),
    PageLocation.FitBH: ('top',),
    PageLocation.FitBV: ('left',),
}
ALL_PAGE_LOCATION_KWARGS = set(chain.from_iterable(PAGE_LOCATION_ARGS.values()))


def make_page_destination(
    pdf: Pdf,
    page_num: int,
    page_location: PageLocation | str | None = None,
    *,
    left: float | None = None,
    top: float | None = None,
    right: float | None = None,
    bottom: float | None = None,
    zoom: float | None = None,
) -> Array:
    """Create a destination ``Array`` with reference to a Pdf document's page number.

    Arguments:
        pdf: PDF document object.
        page_num: Page number (zero-based).
        page_location: Optional page location, as a string or :enum:`PageLocation`.
        left: Specify page viewport rectangle.
        top: Specify page viewport rectangle.
        right: Specify page viewport rectangle.
        bottom: Specify page viewport rectangle.
        zoom: Specify page viewport rectangle's zoom level.

    left, top, right, bottom, zoom are used in conjunction with the page fit style
    specified by *page_location*.
    """
    return _make_page_destination(
        pdf,
        page_num,
        page_location=page_location,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        zoom=zoom,
    )


def _make_page_destination(
    pdf: Pdf,
    page_num: int,
    page_location: PageLocation | str | None = None,
    **kwargs,
) -> Array:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    res: list[Dictionary | Name] = [pdf.pages[page_num].obj]
    if page_location:
        if isinstance(page_location, PageLocation):
            loc_key = page_location
            loc_str = loc_key.name
        else:
            loc_str = page_location
            try:
                loc_key = PageLocation[loc_str]
            except KeyError:
                raise ValueError(
                    f"Invalid or unsupported page location type {loc_str}"
                ) from None
        res.append(Name(f'/{loc_str}'))
        dest_arg_names = PAGE_LOCATION_ARGS.get(loc_key)
        if dest_arg_names:
            res.extend(kwargs.get(k, 0) for k in dest_arg_names)
    else:
        res.append(Name.Fit)
    return Array(res)


class OutlineStructureError(Exception):
    """Indicates an error in the outline data structure."""


class OutlineItem:
    """Manage a single item in a PDF document outlines structure.

    Includes nested items.

    Arguments:
        title: Title of the outlines item.
        destination: Page number, destination name, or any other PDF object
            to be used as a reference when clicking on the outlines entry. Note
            this should be ``None`` if an action is used instead. If set to a
            page number, it will be resolved to a reference at the time of
            writing the outlines back to the document.
        page_location: Supplemental page location for a page number
            in ``destination``, e.g. ``PageLocation.Fit``. May also be
            a simple string such as ``'FitH'``.
        action: Action to perform when clicking on this item. Will be ignored
           during writing if ``destination`` is also set.
        obj: ``Dictionary`` object representing this outlines item in a ``Pdf``.
            May be ``None`` for creating a new object. If present, an existing
            object is modified in-place during writing and original attributes
            are retained.
        left, top, bottom, right, zoom: Describes the viewport position associated
            with a destination.

    This object does not contain any information about higher-level or
    neighboring elements.

    Valid destination arrays:
        [page /XYZ left top zoom]
        generally
        [page, PageLocationEntry, 0 to 4 ints]
    """

    def __init__(
        self,
        title: str,
        destination: Array | String | Name | int | None = None,
        page_location: PageLocation | str | None = None,
        action: Dictionary | None = None,
        obj: Dictionary | None = None,
        *,
        left: float | None = None,
        top: float | None = None,
        right: float | None = None,
        bottom: float | None = None,
        zoom: float | None = None,
    ):
        """Initialize OutlineItem."""
        self.title = title
        self.destination = destination
        self.page_location = page_location
        self.page_location_kwargs = {}
        self.action = action
        if self.destination is not None and self.action is not None:
            raise ValueError("Only one of destination and action may be set")
        self.obj = obj
        kwargs = dict(left=left, top=top, right=right, bottom=bottom, zoom=zoom)
        self.page_location_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.is_closed = False
        self.children: list[OutlineItem] = []

    def __str__(self):
        if self.children:
            if self.is_closed:
                oc_indicator = '[+]'
            else:
                oc_indicator = '[-]'
        else:
            oc_indicator = '[ ]'
        if self.destination is not None:
            if isinstance(self.destination, Array):
                # 12.3.2.2 Explicit destination
                # [raw_page, /PageLocation.SomeThing, integer parameters for viewport]
                raw_page = self.destination[0]
                page = Page(raw_page)
                dest = page.label
            elif isinstance(self.destination, String):
                # 12.3.2.2 Named destination, byte string reference to Names
                dest = (
                    f"<Named Destination in document .Root.Names dictionary: "
                    f"{self.destination}>"
                )
            elif isinstance(self.destination, Name):
                # 12.3.2.2 Named destination, name object (PDF 1.1)
                dest = (
                    f"<Named Destination in document .Root.Dests dictionary: "
                    f"{self.destination}>"
                )
            elif isinstance(self.destination, int):
                # Page number
                dest = f'<Page {self.destination}>'
        else:
            dest = '<Action>'
        return f'{oc_indicator} {self.title} -> {dest}'

    def __repr__(self):
        return f'<pikepdf.{self.__class__.__name__}: "{self.title}">'

    @classmethod
    def from_dictionary_object(cls, obj: Dictionary):
        """Create a ``OutlineItem`` from a ``Dictionary``.

        Does not process nested items.

        Arguments:
            obj: ``Dictionary`` object representing a single outline node.
        """
        title = str(obj.Title)
        destination = obj.get(Name.Dest)
        if destination is not None and not isinstance(
            destination, (Array, String, Name)
        ):
            # 12.3.3: /Dest may be a name, byte string or array
            raise OutlineStructureError(
                f"Unexpected object type in Outline's /Dest: {destination!r}"
            )
        action = obj.get(Name.A)
        if action is not None and not isinstance(action, Dictionary):
            raise OutlineStructureError(
                f"Unexpected object type in Outline's /A: {action!r}"
            )
        return cls(title, destination=destination, action=action, obj=obj)

    def to_dictionary_object(self, pdf: Pdf, create_new: bool = False) -> Dictionary:
        """Create/update a ``Dictionary`` object from this outline node.

        Page numbers are resolved to a page reference on the input
        ``Pdf`` object.

        Arguments:
            pdf: PDF document object.
            create_new: If set to ``True``, creates a new object instead of
                modifying an existing one in-place.
        """
        if create_new or self.obj is None:
            self.obj = obj = pdf.make_indirect(Dictionary())
        else:
            obj = self.obj
        obj.Title = self.title
        if self.destination is not None:
            if isinstance(self.destination, int):
                self.destination = make_page_destination(
                    pdf,
                    self.destination,
                    self.page_location,
                    **self.page_location_kwargs,
                )
            obj.Dest = self.destination
            if Name.A in obj:
                del obj.A
        elif self.action is not None:
            obj.A = self.action
            if Name.Dest in obj:
                del obj.Dest
        return obj


class Outline:
    """Maintains a intuitive interface for creating and editing PDF document outlines.

    See |pdfrm| section 12.3.

    Arguments:
        pdf: PDF document object.
        max_depth: Maximum recursion depth to consider when reading the outline.
        strict: If set to ``False`` (default) silently ignores structural errors.
            Setting it to ``True`` raises a
            :class:`pikepdf.OutlineStructureError`
            if any object references re-occur while the outline is being read or
            written.

    See Also:
        :meth:`pikepdf.Pdf.open_outline`
    """

    def __init__(self, pdf: Pdf, max_depth: int = 15, strict: bool = False):
        """Initialize Outline."""
        self._root: list[OutlineItem] | None = None
        self._pdf = pdf
        self._max_depth = max_depth
        self._strict = strict
        self._updating = False

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return f'<pikepdf.{self.__class__.__name__}: {len(self.root)} items>'

    def __enter__(self):
        self._updating = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                return
            self._save()
        finally:
            self._updating = False

    def _save_level_outline(
        self,
        parent: Dictionary,
        outline_items: Iterable[OutlineItem],
        level: int,
        visited_objs: set[tuple[int, int]],
    ):
        count = 0
        prev: Dictionary | None = None
        first: Dictionary | None = None
        for item in outline_items:
            out_obj = item.to_dictionary_object(self._pdf)
            objgen = out_obj.objgen
            if objgen in visited_objs:
                if self._strict:
                    raise OutlineStructureError(
                        f"Outline object {objgen} reoccurred in structure"
                    )
                out_obj = item.to_dictionary_object(self._pdf, create_new=True)
            else:
                visited_objs.add(objgen)

            out_obj.Parent = parent
            count += 1
            if prev is not None:
                prev.Next = out_obj
                out_obj.Prev = prev
            else:
                first = out_obj
                if Name.Prev in out_obj:
                    del out_obj.Prev
            prev = out_obj
            if level < self._max_depth:
                sub_items: Iterable[OutlineItem] = item.children
            else:
                sub_items = ()
            self._save_level_outline(out_obj, sub_items, level + 1, visited_objs)
            if item.is_closed:
                out_obj.Count = -cast(int, out_obj.Count)
            else:
                count += cast(int, out_obj.Count)
        if count:
            assert prev is not None and first is not None
            if Name.Next in prev:
                del prev.Next
            parent.First = first
            parent.Last = prev
        else:
            if Name.First in parent:
                del parent.First
            if Name.Last in parent:
                del parent.Last
        parent.Count = count

    def _load_level_outline(
        self,
        first_obj: Dictionary,
        outline_items: list[Object],
        level: int,
        visited_objs: set[tuple[int, int]],
    ):
        current_obj: Dictionary | None = first_obj
        while current_obj:
            objgen = current_obj.objgen
            if objgen in visited_objs:
                if self._strict:
                    raise OutlineStructureError(
                        f"Outline object {objgen} reoccurred in structure"
                    )
                return
            visited_objs.add(objgen)

            item = OutlineItem.from_dictionary_object(current_obj)
            first_child = current_obj.get(Name.First)
            if isinstance(first_child, Dictionary) and level < self._max_depth:
                self._load_level_outline(
                    first_child, item.children, level + 1, visited_objs
                )
                count = current_obj.get(Name.Count)
                if isinstance(count, int) and count < 0:
                    item.is_closed = True
            outline_items.append(item)
            next_obj = current_obj.get(Name.Next)
            if next_obj is None or isinstance(next_obj, Dictionary):
                current_obj = next_obj
            else:
                raise OutlineStructureError(
                    f"Outline object {objgen} points to non-dictionary"
                )

    def _save(self):
        if self._root is None:
            return
        if Name.Outlines in self._pdf.Root:
            outlines = self._pdf.Root.Outlines
        else:
            self._pdf.Root.Outlines = outlines = self._pdf.make_indirect(
                Dictionary(Type=Name.Outlines)
            )
        self._save_level_outline(outlines, self._root, 0, set())

    def _load(self):
        self._root = root = []
        if Name.Outlines not in self._pdf.Root:
            return
        outlines = self._pdf.Root.Outlines or {}
        first_obj = outlines.get(Name.First)
        if first_obj:
            self._load_level_outline(first_obj, root, 0, set())

    def add(self, title: str, destination: Array | int | None) -> OutlineItem:
        """Add an item to the outline.

        Arguments:
            title: Title of the outline item.
            destination: Destination to jump to when the item is selected.

        Returns:
            The newly created :class:`OutlineItem`.
        """
        if self._root is None:
            self._load()
        item = OutlineItem(title, destination)
        if self._root is None:
            self._root = [item]
        else:
            self._root.append(item)
        if not self._updating:
            self._save()
        return item

    @property
    def root(self) -> list[OutlineItem]:
        """Return the root node of the outline."""
        if self._root is None:
            self._load()
        return cast(List[OutlineItem], self._root)
