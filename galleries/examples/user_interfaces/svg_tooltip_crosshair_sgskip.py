"""
=====================
SVG crosshair tooltip
=====================

This example shows how to add an interactive crosshair tooltip to an
SVG of line, area, or bar plots. When the user moves the mouse over a
plot, a vertical line tracks the cursor, a coloured dot snaps to each
series' value at that x, and a small table shows the values.

The trick that makes this work for any plot -- not just neatly
normalised time-series data -- is that the data-to-SVG coordinate
transform is computed in Python (via ``Axes.transData``) and embedded
into the SVG as JSON. The JavaScript only has to snap the mouse to the
nearest pre-computed data point, so it doesn't need to know anything
about axis ranges, log scales, date locators, axis padding, or how
matplotlib draws the axes.

Supported artists: ``Line2D`` (``ax.plot``), ``PolyCollection``
(``ax.fill_between``, ``ax.stackplot``), and ``BarContainer``
(``ax.bar``). Multi-axes figures are supported: each series' dot only
moves when the cursor is inside its own axes.

:author: Dylan Jay
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
import json
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ET.register_namespace("", "http://www.w3.org/2000/svg")
_SVG_NS = "http://www.w3.org/2000/svg"


# matplotlib's SVG backend writes the figure in PostScript points
# (72/inch) regardless of figure dpi, so display-pixel coordinates from
# ``ax.transData`` must be scaled to match the SVG viewBox units.
def _disp_to_svg(fig: Figure, disp_xy: np.ndarray) -> np.ndarray:
    """Convert display coords (px, y-up) to SVG viewBox coords (pt, y-down)."""
    scale = 72.0 / float(fig.dpi)
    out = np.empty_like(disp_xy, dtype=float)
    out[..., 0] = disp_xy[..., 0] * scale
    out[..., 1] = (float(fig.bbox.height) - disp_xy[..., 1]) * scale
    return out


def _data_to_svg(fig: Figure, ax: plt.Axes,
                 xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray,
                                                          np.ndarray]:
    """Apply ``ax.transData`` then convert to SVG viewBox coords."""
    svg = _disp_to_svg(fig, ax.transData.transform(
        np.column_stack([xs, ys])))
    return svg[:, 0], svg[:, 1]


def _jsonable(v):
    """Coerce numpy scalars / datetimes to plain Python types for json."""
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def _label_or_empty(artist) -> str:
    """Return the artist's label, or empty if it's matplotlib's default."""
    label = artist.get_label() or ""
    return "" if label.startswith("_") else label


def _series(label: str, colour, xs_data, ys_data,
            svg_x: np.ndarray, svg_y: np.ndarray) -> dict:
    return {
        "label": label,
        "colour": mcolors.to_hex(colour, keep_alpha=False),
        "data_x": [_jsonable(v) for v in xs_data],
        "data_y": [_jsonable(v) for v in ys_data],
        "svg_x": svg_x.tolist(),
        "svg_y": svg_y.tolist(),
    }


def _line_series(fig: Figure, line: Line2D) -> dict:
    """Series dict for a Line2D."""
    # ``orig=False`` returns the float values matplotlib already mapped
    # the data to (e.g. date ordinals), so transData works on them.
    xs = np.asarray(line.get_xdata(orig=False), dtype=float)
    ys = np.asarray(line.get_ydata(orig=False), dtype=float)
    svg_x, svg_y = _data_to_svg(fig, line.axes, xs, ys)
    return _series(_label_or_empty(line), line.get_color(),
                   line.get_xdata(orig=True), line.get_ydata(orig=True),
                   svg_x, svg_y)


def _poly_top_edge(coll: PolyCollection) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) of the upper edge of a filled polygon.

    For each unique x in the polygon's vertices, take the maximum y.
    Works for ``fill_between`` and each ``stackplot`` layer.
    """
    verts = np.concatenate([p.vertices for p in coll.get_paths()])
    order = np.argsort(verts[:, 0], kind="stable")
    xs_sorted = verts[order, 0]
    ys_sorted = verts[order, 1]
    keep = np.concatenate(([True], np.diff(xs_sorted) > 0))
    return xs_sorted[keep], np.maximum.reduceat(ys_sorted,
                                                np.flatnonzero(keep))


def _poly_series(fig: Figure, coll: PolyCollection) -> dict:
    """Series dict for a fill_between / stackplot PolyCollection."""
    xs, ys = _poly_top_edge(coll)
    svg_x, svg_y = _data_to_svg(fig, coll.axes, xs, ys)
    face = np.asarray(coll.get_facecolor())
    colour = face[0] if face.ndim == 2 else face
    return _series(_label_or_empty(coll), colour, xs, ys, svg_x, svg_y)


def _bar_series(fig: Figure, bc: BarContainer) -> dict:
    """Series dict for a BarContainer.

    The dot snaps to the top-center of whichever bar has the closest x
    position. The reported y is the bar's own height -- for stacked
    bars that's the layer's contribution, not the cumulative top.
    """
    xs = np.array([b.get_x() + b.get_width() / 2.0 for b in bc.patches])
    tops = np.array([b.get_y() + b.get_height() for b in bc.patches])
    heights = [b.get_height() for b in bc.patches]
    svg_x, svg_y = _data_to_svg(fig, bc.patches[0].axes, xs, tops)
    return _series(_label_or_empty(bc), bc.patches[0].get_facecolor(),
                   xs, heights, svg_x, svg_y)


def _artist_axes(artist) -> plt.Axes:
    """Return the Axes that owns this artist."""
    if isinstance(artist, BarContainer):
        return artist.patches[0].axes
    return artist.axes


def _axes_bbox(fig: Figure, ax: plt.Axes) -> dict:
    bb = ax.get_window_extent()
    (left, right), (bottom, top) = _disp_to_svg(
        fig, np.array([[bb.xmin, bb.ymin], [bb.xmax, bb.ymax]])).T
    return {"left": float(left), "top": float(top),
            "right": float(right), "bottom": float(bottom)}


def _axes_legend_style(ax: plt.Axes) -> dict:
    """Sample background / border / text colours from this axes' legend,
    falling back to the axes facecolor and rcParams when no legend exists.
    The hover tooltip reuses these so it blends with the current theme.
    """
    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        face, edge = frame.get_facecolor(), frame.get_edgecolor()
        texts = leg.get_texts()
        text_color = texts[0].get_color() if texts else plt.rcParams["text.color"]
    else:
        face = ax.get_facecolor()
        edge = plt.rcParams["axes.edgecolor"]
        text_color = plt.rcParams["text.color"]
    return {"background": mcolors.to_hex(face, keep_alpha=True),
            "border": mcolors.to_hex(edge, keep_alpha=False),
            "color": mcolors.to_hex(text_color, keep_alpha=False)}


# Static styling. Per-axes colours are applied at hover time from
# ``axes_bboxes[i].style`` (see ``_axes_legend_style``).
_TOOLTIP_STYLE = ("border:1px solid; padding:6px 8px; "
                  "display:inline-block; "
                  "font:11px 'DejaVu Sans', sans-serif;")


def _overlay_svg(series: Sequence[dict]) -> ET.Element:
    """Build the hidden tooltip/line/dots SVG group."""
    dots = "".join(
        f'<circle id="dot_{i}" r="5" fill="{s["colour"]}" cx="0" cy="0"/>'
        for i, s in enumerate(series))
    xml = f"""
    <g xmlns="{_SVG_NS}" id="hover_overlay" visibility="hidden"
       pointer-events="none">
      <line id="hover_line" x1="0" y1="0" x2="0" y2="0"
            stroke="#808080" stroke-dasharray="3.7,1.6"/>
      {dots}
      <foreignObject id="hover_tooltip" width="300" height="200"
                     overflow="visible" x="0" y="0">
        <body xmlns="http://www.w3.org/1999/xhtml">
          <div id="hover_tooltip_box" style="{_TOOLTIP_STYLE}">
            <table id="hover_table"></table>
          </div>
        </body>
      </foreignObject>
    </g>
    """
    return ET.fromstring(xml)


_JS_TEMPLATE = r"""
(() => {
    const SERIES = __SERIES__, AXES = __AXES__;
    const XHTML = "http://www.w3.org/1999/xhtml";
    const svg = document.documentElement;
    const overlay = document.getElementById("hover_overlay");
    const line = document.getElementById("hover_line");
    const tooltip = document.getElementById("hover_tooltip");
    const tooltipBox = document.getElementById("hover_tooltip_box");
    const table = document.getElementById("hover_table");
    const dots = SERIES.map((_, i) => document.getElementById("dot_" + i));

    const argmin = (arr, x) => arr.reduce(
        (best, v, i, a) =>
            Math.abs(v - x) < Math.abs(a[best] - x) ? i : best, 0);
    const fmt = v => (typeof v === "number" && !Number.isInteger(v))
        ? v.toFixed(3) : String(v ?? "");
    const td = (text, style) => {
        const el = document.createElementNS(XHTML, "td");
        el.setAttribute("style", style); el.textContent = text; return el;
    };

    svg.addEventListener("mousemove", evt => {
        // Mouse position in the SVG's own viewBox coordinate system.
        const ctm = svg.getScreenCTM();
        if (!ctm) return;
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX; pt.y = evt.clientY;
        const { x: mx, y: my } = pt.matrixTransform(ctm.inverse());

        // Which axes is the mouse over?
        const ax = AXES.find(b => mx >= b.left && mx <= b.right
                                  && my >= b.top && my <= b.bottom);
        if (!ax) {
            overlay.setAttribute("visibility", "hidden");
            setLegends("visible");
            return;
        }

        // Apply this axes' legend styling to the tooltip box.
        tooltipBox.style.background = ax.style.background;
        tooltipBox.style.borderColor = ax.style.border;
        tooltipBox.style.color = ax.style.color;

        // Hide the active axes' legend; restore others.
        for (const a of AXES) {
            if (!a.legend_gid) continue;
            document.getElementById(a.legend_gid).setAttribute(
                "visibility", a === ax ? "hidden" : "visible");
        }
        overlay.setAttribute("visibility", "visible");

        // Vertical cursor line spans the active axes.
        line.setAttribute("x1", mx); line.setAttribute("x2", mx);
        line.setAttribute("y1", ax.top); line.setAttribute("y2", ax.bottom);

        // Show only series in the active axes; snap each to its nearest point.
        table.replaceChildren();
        SERIES.forEach((s, i) => {
            if (s.axes_idx !== ax.idx) {
                dots[i].setAttribute("visibility", "hidden"); return;
            }
            dots[i].setAttribute("visibility", "visible");
            const k = argmin(s.svg_x, mx);
            dots[i].setAttribute("cx", s.svg_x[k]);
            dots[i].setAttribute("cy", s.svg_y[k]);
            const tr = document.createElementNS(XHTML, "tr");
            const right = "text-align:right;padding-right:8px;";
            tr.appendChild(td(s.label,
                              "color:" + s.colour + ";padding-right:8px;"));
            tr.appendChild(td(fmt(s.data_x[k]), right));
            tr.appendChild(td(fmt(s.data_y[k]), "text-align:right;"));
            table.appendChild(tr);
        });

        // Place tooltip: prefer right/below cursor; flip and clamp so
        // it stays inside the figure (SVG viewBox), not just the axes.
        // Use the inner div's rect because the foreignObject's declared
        // width/height don't shrink-wrap to content.
        const GAP = 12;
        const box = tooltipBox.getBoundingClientRect();
        const fw = box.width / ctm.a, fh = box.height / ctm.d;
        const vb = svg.viewBox.baseVal;
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
        let tx = mx + GAP;
        if (tx + fw > vb.x + vb.width) tx = mx - GAP - fw;
        tx = clamp(tx, vb.x, vb.x + vb.width - fw);
        let ty = my + GAP;
        if (ty + fh > vb.y + vb.height) ty = my - GAP - fh;
        ty = clamp(ty, vb.y, vb.y + vb.height - fh);
        tooltip.setAttribute("x", tx); tooltip.setAttribute("y", ty);
    });

    svg.addEventListener("mouseleave", () => {
        overlay.setAttribute("visibility", "hidden");
        setLegends("visible");
    });

    function setLegends(visibility) {
        for (const a of AXES) {
            if (!a.legend_gid) continue;
            document.getElementById(a.legend_gid)
                .setAttribute("visibility", visibility);
        }
    }
})();
"""


def add_crosshair_tooltip(fig: Figure, path: str,
                          artists=None) -> None:
    """Save ``fig`` as an SVG with an interactive crosshair tooltip.

    When a viewer moves the mouse over the plot, a vertical line tracks
    the cursor, a coloured dot snaps to each series' nearest data point,
    and a small table shows the values.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing one or more Axes with Line2D, fill_between /
        stackplot polygons, or bar charts.
    path : str
        Where to write the resulting SVG file.
    artists : sequence of Line2D, PolyCollection, or BarContainer, optional
        Restrict the tooltip to these artists. If omitted, every
        supported artist in every axes of ``fig`` is included in draw
        order.
    """
    fig.canvas.draw()

    if artists is None:
        artists = []
        for ax in fig.axes:
            artists.extend(ax.get_lines())
            artists.extend(c for c in ax.collections
                           if isinstance(c, PolyCollection))
            artists.extend(c for c in ax.containers
                           if isinstance(c, BarContainer))
    if not artists:
        raise ValueError(
            "no Line2D, PolyCollection or BarContainer artists in figure")

    # Stable index per axes (used to scope each series to its own axes
    # so its dot only moves when the cursor is inside that axes).
    fig_axes = list(fig.axes)
    series = []
    for artist in artists:
        if isinstance(artist, Line2D):
            s = _line_series(fig, artist)
        elif isinstance(artist, PolyCollection):
            s = _poly_series(fig, artist)
        elif isinstance(artist, BarContainer):
            s = _bar_series(fig, artist)
        else:
            raise TypeError(
                f"unsupported artist type: {type(artist).__name__}")
        s["axes_idx"] = fig_axes.index(_artist_axes(artist))
        series.append(s)
    axes_bboxes = []
    seen = set()
    for artist in artists:
        ax = _artist_axes(artist)
        if id(ax) not in seen:
            seen.add(id(ax))
            bbox = _axes_bbox(fig, ax)
            bbox["idx"] = fig_axes.index(ax)
            bbox["style"] = _axes_legend_style(ax)
            leg = ax.get_legend()
            if leg is not None:
                # Give the legend a stable gid (unless the user set one)
                # so the JS can hide it during hover.
                leg.set_gid(leg.get_gid() or f"hover_legend_{bbox['idx']}")
                bbox["legend_gid"] = leg.get_gid()
            else:
                bbox["legend_gid"] = None
            axes_bboxes.append(bbox)

    buf = BytesIO()
    fig.savefig(buf, format="svg")
    tree, _xmlid = ET.XMLID(buf.getvalue())

    tree.append(_overlay_svg(series))

    script_body = (_JS_TEMPLATE
                   .replace("__SERIES__", json.dumps(series))
                   .replace("__AXES__", json.dumps(axes_bboxes)))
    script = ET.SubElement(tree, f"{{{_SVG_NS}}}script",
                           {"type": "text/ecmascript"})
    # Wrap in CDATA so JS containing ``<`` / ``&`` parses correctly.
    script.text = f"// <![CDATA[\n{script_body}\n// ]]>"

    ET.ElementTree(tree).write(path, xml_declaration=True, encoding="utf-8")


# --- demo -------------------------------------------------------------------

def _demo() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 40)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(x, np.sin(x) * 20 + 30, label="sine")
    axs[0, 0].plot(x, np.cos(x) * 15 + 25, label="cosine")
    axs[0, 0].plot(x, rng.normal(size=x.size).cumsum() + 30,
                   label="random walk")
    axs[0, 0].set_title("lines")
    axs[0, 0].legend()

    axs[0, 1].fill_between(x, np.sin(x) * 20 + 30, alpha=0.4, label="area")
    axs[0, 1].set_title("fill_between")
    axs[0, 1].legend()

    axs[1, 0].stackplot(x,
                        np.abs(np.sin(x)) * 10 + 5,
                        np.abs(np.cos(x)) * 15 + 5,
                        labels=["layer 1", "layer 2"])
    axs[1, 0].set_title("stackplot")
    axs[1, 0].legend()

    axs[1, 1].bar(range(10), rng.integers(5, 30, size=10), label="counts")
    axs[1, 1].set_title("bar")
    axs[1, 1].legend()

    fig.tight_layout()
    add_crosshair_tooltip(fig, "svg_tooltip_crosshair.svg")
    print("wrote svg_tooltip_crosshair.svg")


if __name__ == "__main__":
    _demo()
