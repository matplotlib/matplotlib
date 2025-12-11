from .description import Desc, desc_like
from .conversion_edge import Graph, TransformEdge


def containerize_draw(draw_func):
    def draw(self, renderer, *, graph=None):
        if graph is None:
            graph = Graph([])

        implicit_graph = _get_graph(self.axes)
        return draw_func(self, renderer, graph=graph+implicit_graph)

    return draw


def _get_graph(ax):
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
