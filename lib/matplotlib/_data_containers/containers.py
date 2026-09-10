from __future__ import annotations

from typing import (
    Protocol,
    Optional,
    Any,
    Union,
)
from collections.abc import Callable, MutableMapping
import uuid

from cachetools import LFUCache  # type: ignore[import-untyped]

import numpy as np

from .description import Desc, desc_like

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversion_edge import Graph


class _MatplotlibTransform(Protocol):
    def transform(self, verts): ...

    def __sub__(self, other) -> "_MatplotlibTransform": ...


class DataContainer(Protocol):
    def query(
        self,
        graph: Graph,
        parent_coordinates: str = "axes",
        /,
    ) -> tuple[dict[str, Any], Union[str, int]]:
        """
        Query the data container for data.

        We are given the data limits and the screen size so that we have an
        estimate of how finely (or not) we need to sample the data we wrapping.

        Parameters
        ----------
        coord_transform : matplotlib.transform.Transform
            Must go from axes fraction space -> data space

        size : 2 integers
            xpixels, ypixels

            The size in screen / render units that we have to fill.

        Returns
        -------
        data : dict[str, Any]
            The values are really array-likes

        cache_key : str
            This is a key that clients can use to cache down-stream
            computations on this data.
        """
        ...

    def describe(self) -> dict[str, Desc]:
        """
        Describe the data a query will return

        Returns
        -------
        dict[str, Desc]
        """
        ...


class NoNewKeys(ValueError): ...


class ArrayContainer:
    def __init__(self, coordinates: dict[str, str] | None = None, /, **data):
        coordinates = coordinates or {}
        self._data = data
        self._cache_key = str(uuid.uuid4())
        self._desc = {
            k: (
                Desc(v.shape, coordinates.get(k, "auto"))
                if hasattr(v, "shape")
                else Desc((), coordinates.get(k, "auto"))
            )
            for k, v in data.items()
        }

    def query(
        self,
        graph: Graph,
        parent_coordinates: str = "axes",
    ) -> tuple[dict[str, Any], Union[str, int]]:
        return dict(self._data), self._cache_key

    def describe(self) -> dict[str, Desc]:
        return dict(self._desc)

    def update(self, **data):
        # TODO check that this is still consistent with desc!
        if not all(k in self._data for k in data):
            raise NoNewKeys(
                f"The keys that currently exist are {set(self._data)}.  You "
                f"tried to add {set(data) - set(self._data)!r}."
            )
        self._data.update(data)
        self._cache_key = str(uuid.uuid4())


class FuncContainer:
    def __init__(
        self,
        # TODO: is this really the best spelling?!
        xfuncs: Optional[
            dict[str, tuple[tuple[Union[str, int], ...], Callable[[Any], Any]]]
        ] = None,
        yfuncs: Optional[
            dict[str, tuple[tuple[Union[str, int], ...], Callable[[Any], Any]]]
        ] = None,
        xyfuncs: Optional[
            dict[str, tuple[tuple[Union[str, int], ...], Callable[[Any, Any], Any]]]
        ] = None,
    ):
        """
        A container that wraps several functions.  They are split into 3 categories:

          - functions that are offered x-like values as input
          - functions that are offered y-like values as input
          - functions that are offered both x and y like values as two inputs

        In addition to the callable, the user needs to provide a spelling of
        what the (relative) shapes will be in relation to each other. For now this
        is a list of integers and strings, where the strings are "generic" values.

        For example if two functions report shapes: ``{'bins':[N],  'edges': [N + 1]``
        then when called, *edges* will always have one more entry than bins.

        Parameters
        ----------
        xfuncs, yfuncs, xyfuncs : dict[str, tuple[shape, func]]

        """
        self._desc: dict[str, Desc] = {}

        def _split(input_dict):
            out = {}
            for k, (shape, func) in input_dict.items():
                self._desc[k] = Desc(shape)
                out[k] = func
            return out

        self._xfuncs = _split(xfuncs) if xfuncs is not None else {}
        self._yfuncs = _split(yfuncs) if yfuncs is not None else {}
        self._xyfuncs = _split(xyfuncs) if xyfuncs is not None else {}
        self._cache: MutableMapping[Union[str, int], Any] = LFUCache(64)

    def _query_hash(self, data_lim, size):
        xlims, ylims = data_lim.evaluate({"x": [0, 1], "y": [0, 1]}).values()
        data_bounds = (*(float(x) for x in xlims), *(float(y) for y in ylims))
        hash_key = hash((data_bounds, size))
        return hash_key

    def query(
        self,
        graph: Graph,
        parent_coordinates: str = "axes",
    ) -> tuple[dict[str, Any], Union[str, int]]:
        desc = Desc(("N",))
        xy = {"x": desc, "y": desc}
        data_lim = graph.evaluator(
            desc_like(xy, coordinates="data"),
            desc_like(xy, coordinates=parent_coordinates),
        ).inverse

        screen_size = graph.evaluator(
            desc_like(xy, coordinates=parent_coordinates),
            desc_like(xy, coordinates="display"),
        )

        screen_dims = screen_size.evaluate({"x": [0, 1], "y": [0, 1]})
        xpix, ypix = np.ceil(np.abs(np.diff(screen_dims["x"]))), np.ceil(
            np.abs(np.diff(screen_dims["y"]))
        )
        xpix = int(xpix)
        ypix = int(ypix)

        hash_key = self._query_hash(data_lim, (xpix, ypix))
        if hash_key in self._cache:
            return self._cache[hash_key], hash_key

        x_data = data_lim.evaluate(
            {
                "x": np.linspace(0, 1, xpix * 2),
                "y": np.zeros(xpix * 2),
            }
        )["x"]
        y_data = data_lim.evaluate(
            {
                "x": np.zeros(ypix * 2),
                "y": np.linspace(0, 1, ypix * 2),
            }
        )["y"]

        ret = self._cache[hash_key] = dict(
            **{k: f(x_data) for k, f in self._xfuncs.items()},
            **{k: f(y_data) for k, f in self._yfuncs.items()},
            **{k: f(x_data, y_data) for k, f in self._xyfuncs.items()},
        )
        return ret, hash_key

    def describe(self) -> dict[str, Desc]:
        return dict(self._desc)
