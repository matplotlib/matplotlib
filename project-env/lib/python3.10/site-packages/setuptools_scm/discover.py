from __future__ import annotations

import os
from typing import Iterable
from typing import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import _types as _t
from .config import Configuration
from .utils import trace


def walk_potential_roots(
    root: _t.PathT, search_parents: bool = True
) -> Iterator[_t.PathT]:
    """
    Iterate though a path and each of its parents.
    :param root: File path.
    :param search_parents: If ``False`` the parents are not considered.
    """

    if not search_parents:
        yield root
        return

    tail = root

    while tail:
        yield root
        root, tail = os.path.split(root)


def match_entrypoint(root: _t.PathT, name: str) -> bool:
    """
    Consider a ``root`` as entry-point.
    :param root: File path.
    :param name: Subdirectory name.
    :return: ``True`` if a subdirectory ``name`` exits in ``root``.
    """

    if os.path.exists(os.path.join(root, name)):
        if not os.path.isabs(name):
            return True
        trace("ignoring bad ep", name)

    return False


def iter_matching_entrypoints(
    root: _t.PathT, entrypoint: str, config: Configuration
) -> Iterable[_t.EntrypointProtocol]:
    """
    Consider different entry-points in ``root`` and optionally its parents.
    :param root: File path.
    :param entrypoint: Entry-point to consider.
    :param config: Configuration,
        read ``search_parent_directories``, write found parent to ``parent``.
    """

    trace("looking for ep", entrypoint, root)
    from ._entrypoints import iter_entry_points

    for wd in walk_potential_roots(root, config.search_parent_directories):
        for ep in iter_entry_points(entrypoint):
            if match_entrypoint(wd, ep.name):
                trace("found ep", ep, "in", wd)
                config.parent = wd
                yield ep
