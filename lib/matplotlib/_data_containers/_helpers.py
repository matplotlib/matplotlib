from .description import Desc, desc_like
from .conversion_edge import Graph, TransformEdge


def _get_graph(ax):
    """Compute the Graph for a given axes.

    Produces a minimal graph that provides enough for `FuncContainer`.
    """
    if ax is None:
        return Graph([])
    desc: Desc = Desc(("N",), coordinates="data")
    xy: dict[str, Desc] = {"x": desc, "y": desc}
    implicit_graph = Graph(
        [
            TransformEdge(
                "data",
                xy,
                desc_like(xy, coordinates="axes"),
                transform=ax.transData - ax.transAxes,
            ),
            TransformEdge(
                "axes",
                desc_like(xy, coordinates="axes"),
                desc_like(xy, coordinates="display"),
                transform=ax.transAxes,
            ),
            TransformEdge(
                "dpi",
                desc_like(xy, coordinates="display_inches"),
                desc_like(xy, coordinates="display"),
                transform=ax.figure.dpi_scale_trans,
            ),
        ],
        aliases=(("parent", "axes"),),
    )
    return implicit_graph


def check_container(artist, container_cls, operation="This operation"):
    """Validation helper for backwards compatibility checks"""
    if not isinstance(artist._container, container_cls):
        raise TypeError(f"{operation} is not available with a custom container class")
