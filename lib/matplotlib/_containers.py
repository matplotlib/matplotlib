from mpl_data_containers.description import Desc, desc_like
from mpl_data_containers.conversion_edge import Graph, TransformEdge


def containerize_draw(draw_func):
    def draw(self, renderer, *, graph=None):
        if graph is None:
            graph = Graph([])

        ax = self.axes
        if ax is None:
            implicit_graph = Graph([])
        else:
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
                ],
                aliases=(("parent", "axes"),),
            )

        return draw_func(self, renderer, graph=graph+implicit_graph)

    return draw
