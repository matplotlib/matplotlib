from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Callable
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Any
import numpy as np

from .description import Desc, desc_like, ShapeSpec

from matplotlib.transforms import Transform


@dataclass
class Edge:
    name: str
    input: dict[str, Desc]
    output: dict[str, Desc]
    weight: float = 1
    invertable: bool = True

    def evaluate(self, input: dict[str, Any]) -> dict[str, Any]:
        return input

    @property
    def inverse(self) -> "Edge":
        return Edge(self.name + "_r", self.output, self.input, self.weight)


@dataclass
class SequenceEdge(Edge):
    edges: Sequence[Edge] = ()

    @classmethod
    def from_edges(
        cls,
        name: str,
        edges: Sequence[Edge],
        output: dict[str, Desc],
        weight: float | None = None,
    ):
        input: dict[str, Desc] = {}
        intermediates: dict[str, Desc] = {}
        invertable = True
        edge_sum: float = 0
        for edge in edges:
            edge_sum += edge.weight
            input |= {k: v for k, v in edge.input.items() if k not in intermediates}
            intermediates |= edge.output
            if not edge.invertable:
                invertable = False

        if weight is None:
            weight = edge_sum

        return cls(name, input, output, weight, invertable, edges)

    def evaluate(self, input: dict[str, Any]) -> dict[str, Any]:
        for edge in self.edges:
            input |= edge.evaluate({k: input[k] for k in edge.input})
        return {k: input[k] for k in self.output}

    @property
    def inverse(self) -> "SequenceEdge":
        return SequenceEdge.from_edges(
            self.name + "_r",
            [e.inverse for e in self.edges[::-1]],
            self.input,
            self.weight,
        )


@dataclass
class CoordinateEdge(Edge):
    """Change coordinates without changing values"""

    @classmethod
    def from_coords(
        cls, name: str, input: dict[str, Desc | str], output: str, weight: float = 1
    ):
        # dtype/shape is reductive here, but I like the idea of being able to just
        # supply only the input/output coordinates for many things
        # could also see lowering default weight for this edge, but just defaulting
        # everything to 1 for now
        inp = {
            k: v if isinstance(v, Desc) else Desc(("N",), v) for k, v in input.items()
        }
        outp = {k: desc_like(v, coordinates=output) for k, v in inp.items()}

        return cls(name, inp, outp, weight)

    @property
    def inverse(self) -> Edge:
        return Edge(f"{self.name}_r", self.output, self.input, self.weight)


@dataclass
class DefaultEdge(Edge):
    """Provide default values with a high weight"""

    weight = 1e6
    value: Any = None

    @classmethod
    def from_default_value(
        cls,
        name: str,
        key: str,
        output: Desc,
        value: Any,
        weight=1e6,
    ) -> "DefaultEdge":
        return cls(name, {}, {key: output}, weight, invertable=False, value=value)

    @classmethod
    def from_rc(
        cls, rc_name: str, key: str | None = None, coordinates: str = "display"
    ):
        from matplotlib import rcParams

        if key is None:
            key = rc_name.split(".")[-1]
        scalar = Desc((), coordinates)
        return cls.from_default_value(f"{rc_name}_rc", key, scalar, rcParams[rc_name])

    def evaluate(self, input: dict[str, Any]) -> dict[str, Any]:
        return {k: self.value for k in self.output}


@dataclass
class FuncEdge(Edge):
    # TODO: more explicit callable boundaries?
    func: Callable = lambda: {}
    inverse_func: Callable | None = None

    @classmethod
    def from_func(
        cls,
        name: str,
        func: Callable,
        input: str | dict[str, Desc],
        output: str | dict[str, Desc],
        weight: float = 1,
        inverse: Callable | None = None,
    ):
        # dtype/shape is reductive here, but I like the idea of being able to just
        # supply a function and the input/output coordinates for many things
        if isinstance(input, str):
            import inspect

            input_vars = inspect.signature(func).parameters.keys()
            input = {k: Desc(("N",), input) for k in input_vars}
        if isinstance(output, str):
            output = {k: Desc(("N",), output) for k in input.keys()}

        return cls(name, input, output, weight, inverse is not None, func, inverse)

    def evaluate(self, input: dict[str, Any]) -> dict[str, Any]:
        res = self.func(**{k: input[k] for k in self.input})

        if isinstance(res, dict):
            # TODO: more sanity checks here?
            # How forgiving do we _really_ wish to be?
            return res
        elif isinstance(res, tuple):
            if len(res) != len(self.output):
                if len(self.output) == 1:
                    return {k: res for k in self.output}
                raise RuntimeError(
                    f"Expected {len(self.output)} return values,"
                    f"got {len(res)} in {self.name}"
                )
            return {k: v for k, v in zip(self.output, res)}
        elif len(self.output) == 1:
            return {k: res for k in self.output}
        raise RuntimeError("Output of function does not match expected output")

    @property
    def inverse(self) -> "FuncEdge":
        if self.inverse_func is None:
            raise RuntimeError("Trying to invert a non-invertable edge")

        return FuncEdge.from_func(
            self.name + "_r",
            self.inverse_func,
            self.output,
            self.input,
            self.weight,
            self.func,
        )


@dataclass
class TransformEdge(Edge):
    transform: Transform | Callable[[], Transform] | None = None

    # TODO: helper for common cases/validation?

    def evaluate(self, input: dict[str, Any]) -> dict[str, Any]:
        # TODO: ensure ordering?
        # Stacking and unstacking at every step seems inefficient,
        # especially if initially given as stacked
        if self.transform is None:
            return input
        elif isinstance(self.transform, Callable):   # type: ignore[arg-type]
            trf = self.transform()                   # type: ignore[operator]
        else:
            trf = self.transform
        inp = np.stack([input[k] for k in self.input], axis=-1)
        outp = trf.transform(inp)
        return {k: v for k, v in zip(self.output, outp.T)}

    @property
    def inverse(self) -> "TransformEdge":
        if self.transform is None:
            raise RuntimeError("Trying to invert a non-invertable edge")

        if isinstance(self.transform, Callable):   # type: ignore[arg-type]
            return TransformEdge(
                self.name + "_r",
                self.output,
                self.input,
                self.weight,
                True,
                lambda: self.transform().inverted(),  # type: ignore[misc,operator]
            )

        return TransformEdge(
            self.name + "_r",
            self.output,
            self.input,
            self.weight,
            True,
            self.transform.inverted(),  # type: ignore[union-attr]
        )


class Graph:
    def __init__(
        self, edges: Sequence[Edge], aliases: tuple[tuple[str, str], ...] = ()
    ):
        self._edges = tuple(edges)
        self._aliases = aliases

        self._subgraphs: list[tuple[set[str], list[Edge]]] = []
        for edge in self._edges:
            keys = set(edge.input) | set(edge.output)

            overlapping = []

            for n, (sub_keys, sub_edges) in enumerate(self._subgraphs):
                if keys & sub_keys:
                    overlapping.append(n)

            if not overlapping:
                self._subgraphs.append((keys, [edge]))
            elif len(overlapping) == 1:
                s = self._subgraphs[overlapping[0]][0]
                s |= keys
                self._subgraphs[overlapping[0]][1].append(edge)
            else:
                edges_combined = [edge]
                for n in overlapping:
                    keys |= self._subgraphs[n][0]
                    edges_combined.extend(self._subgraphs[n][1])
                for n in overlapping[::-1]:
                    self._subgraphs.pop(n)
                self._subgraphs.append((keys, edges_combined))

    def _resolve_alias(self, coord: str) -> str:
        while True:
            for coa, cob in self._aliases:
                if coord == coa:
                    coord = cob
                    break
            else:
                break
        return coord

    def evaluator(self, input: dict[str, Desc], output: dict[str, Desc]) -> Edge:
        out_edges = []

        for sub_keys, sub_edges in self._subgraphs:
            if not (sub_keys & set(output) or sub_keys & set(input)):
                continue

            output_subset = {k: v for k, v in output.items() if k in sub_keys}
            sub_edges = sorted(sub_edges, key=lambda x: x.weight)

            @dataclass
            class Node:
                weight: float
                desc: dict[str, Desc]
                prev_node: Node | None = None
                edge: Edge | None = None

                def __le__(self, other):
                    return self.weight <= other.weight

                def __lt__(self, other):
                    return self.weight < other.weight

                def __ge__(self, other):
                    return self.weight >= other.weight

                def __gt__(self, other):
                    return self.weight > other.weight

                @property
                def edges(self):
                    if self.prev_node is None:
                        return [self.edge]
                    return self.prev_node.edges + [self.edge]

            q: PriorityQueue[Node] = PriorityQueue()
            q.put(Node(0, input))

            best: Node = Node(np.inf, {})
            while not q.empty():
                n = q.get()
                if n.weight > best.weight:
                    continue
                if Desc.compatible(n.desc, output_subset, aliases=self._aliases):
                    if n.weight < best.weight:
                        best = n
                    continue
                for e in sub_edges:
                    if e in n.edges:
                        continue
                    if Desc.compatible(n.desc, e.input, aliases=self._aliases):
                        d = n.desc | e.output
                        w = n.weight + e.weight

                        q.put(Node(w, d, n, e))
            if np.isinf(best.weight):
                raise NotImplementedError(
                    "This may be possible, but is not a simple case already considered"
                )

            edges: list[Edge] = []
            n = best
            while n.prev_node is not None:
                if n.edge is not None:
                    edges.insert(0, n.edge)
                n = n.prev_node
            if len(edges) == 0:
                continue
            elif len(edges) == 1:
                out_edges.append(edges[0])
            else:
                out_edges.append(SequenceEdge.from_edges("eval", edges, output_subset))

        found_outputs = set(input)
        for out in out_edges:
            found_outputs |= set(out.output)
        if missing := set(output) - found_outputs:
            raise RuntimeError(f"Could not find path to resolve all outputs: {missing}")

        if len(out_edges) == 0:
            return Edge("noop", input, output)
        if len(out_edges) == 1:
            return out_edges[0]
        return SequenceEdge.from_edges("eval", out_edges, output)

    def __add__(self, other: Graph) -> Graph:
        aself = {k: v for k, v in self._aliases}
        aother = {k: v for k, v in other._aliases}
        aliases = tuple((aself | aother).items())
        return Graph(self._edges + other._edges, aliases)

    def cache_key(self):
        """A cache key representing the graph.

        Current implementation is a new UUID, that is to say uncachable.
        """
        import uuid

        return str(uuid.uuid4())


def coord_and_default(
    key: str,
    shape: ShapeSpec = (),
    coordinates: str = "display",
    default_value: Any = None,
    default_rc: str | None = None,
):
    if default_rc is not None:
        if default_value is not None:
            raise ValueError(
                "Only one of 'default_value' and 'default_rc' may be specified"
            )
        def_edge = DefaultEdge.from_rc(default_rc, key, coordinates)
    else:
        scalar = Desc((), coordinates)
        def_edge = DefaultEdge.from_default_value(
            f"{key}_def", key, scalar, default_value
        )
    coord_edge = CoordinateEdge.from_coords(key, {key: Desc(shape)}, coordinates)
    return coord_edge, def_edge
