"""Tests for the SVG line/area tooltip example.

These tests render a matplotlib figure with the hover overlay applied,
load the SVG in a headless browser via Playwright, simulate a mousemove,
and verify the resulting tooltip dot lies on the **matplotlib-drawn
SVG path** for its series. The path is matplotlib's own rendering of
the line/area, so it's an oracle that is independent of the example's
data-to-SVG transform code: if our scaling were wrong, the dot would
land off the rendered line and the test would fail.

Each plotted artist is given a stable ``gid`` (``series_0``,
``series_1``, ...) so the test can locate its ``<path>`` in the SVG.

Playwright is a soft dependency -- the whole module is skipped if it's
missing.
"""
from __future__ import annotations

import contextlib
import pathlib
import sys

import pytest

import matplotlib.pyplot as plt

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

svg_tooltip_crosshair = pytest.importorskip("svg_tooltip_crosshair_sgskip")


# --- browser-side helpers ---------------------------------------------------

# Walk a ``<path>`` inside the SVG element with id ``gid`` and return the
# minimum y (= topmost point in screen-y terms) at exactly ``target_x``
# along the path. The path may cross ``target_x`` more than once (a
# fill_between polygon visits each x on its top edge and on its bottom
# edge); we bisect each crossing precisely and return the topmost.
_TOP_Y_AT_X_JS = """([gid, target_x]) => {
    const group = document.getElementById(gid);
    if (!group) { throw new Error('no element with id ' + gid); }
    const path = (group.tagName.toLowerCase() === 'path')
        ? group : group.querySelector('path');
    if (!path) { throw new Error('no <path> inside #' + gid); }
    const len = path.getTotalLength();
    const N = 400;
    const crossings = [];
    let prev = path.getPointAtLength(0);
    if (Math.abs(prev.x - target_x) < 1e-9) {
        crossings.push([0, 0]);
    }
    for (let i = 1; i <= N; i++) {
        const t = (i / N) * len;
        const pt = path.getPointAtLength(t);
        const a = prev.x - target_x;
        const b = pt.x - target_x;
        if (a !== 0 && b !== 0 && a * b < 0) {
            crossings.push([(i - 1) / N * len, t]);
        } else if (b === 0) {
            crossings.push([t, t]);
        }
        prev = pt;
    }
    if (crossings.length === 0) { return null; }
    const ys = crossings.map(([lo0, hi0]) => {
        let lo = lo0, hi = hi0;
        for (let k = 0; k < 40; k++) {
            const mid = (lo + hi) / 2;
            const xm = path.getPointAtLength(mid).x;
            if ((path.getPointAtLength(lo).x - target_x)
                * (xm - target_x) <= 0) { hi = mid; }
            else { lo = mid; }
        }
        return path.getPointAtLength((lo + hi) / 2).y;
    });
    return Math.min.apply(null, ys);
}"""


@contextlib.contextmanager
def _open(svg_path: pathlib.Path, viewport=(800, 600)):
    """Open ``svg_path`` in a fresh chromium page; yield the page."""
    w, h = viewport
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(svg_path.as_uri())
            page.wait_for_load_state("networkidle")
            yield page
        finally:
            browser.close()


def _axes_client_bbox(page, axes_gid: str = "axes_1") -> dict:
    """matplotlib wraps each Axes in ``<g id="axes_N">``. Return that
    group's bounding box in CSS pixels.
    """
    bb = page.locator(f"#{axes_gid}").bounding_box()
    if bb is None:
        raise RuntimeError(f"could not locate axes group #{axes_gid}")
    return bb


def _hover_at_axes_fraction(page, fx: float, fy: float,
                            axes_gid: str = "axes_1") -> None:
    """Hover at the (fx, fy) fraction of the axes bounding box. fx/fy
    outside [0, 1] place the cursor outside the axes.
    """
    bb = _axes_client_bbox(page, axes_gid)
    page.mouse.move(bb["x"] + bb["width"] * fx,
                    bb["y"] + bb["height"] * fy)
    page.wait_for_timeout(100)


def _attr_float(page, sel: str, attr: str) -> float | None:
    v = page.locator(sel).get_attribute(attr)
    return float(v) if v is not None else None


def _assert_dot_on_series(page, dot_id: str, series_gid: str) -> None:
    """Read dot's (cx, cy) and assert it lies on the matplotlib-drawn
    path identified by ``series_gid``.
    """
    cx = _attr_float(page, f"#{dot_id}", "cx")
    cy = _attr_float(page, f"#{dot_id}", "cy")
    assert cx is not None and cy is not None, (
        f"#{dot_id} has no cx/cy -- overlay may not be visible")
    expected_y = page.evaluate(_TOP_Y_AT_X_JS, [series_gid, cx])
    assert expected_y is not None, (
        f"could not find path point at x={cx} on #{series_gid}")
    assert cy == pytest.approx(expected_y, abs=1.5), (
        f"#{dot_id} at ({cx:.1f}, {cy:.1f}) is not on #{series_gid} "
        f"(path y at that x = {expected_y:.1f})")


# --- plot builders ----------------------------------------------------------

def _save(fig, tmp_path) -> pathlib.Path:
    fig.canvas.draw()
    svg_path = tmp_path / "plot.svg"
    svg_tooltip_crosshair.add_crosshair_tooltip(fig, str(svg_path))
    return svg_path


def _line_plot(tmp_path, xs, ys, *, legend: bool = True):
    """Single Line2D with ``gid='series_0'``. Returns (fig, ax, svg_path)."""
    fig, ax = plt.subplots()
    line, = ax.plot(xs, ys, label="s")
    line.set_gid("series_0")
    if legend:
        ax.legend()
    return fig, ax, _save(fig, tmp_path)


# --- tests ------------------------------------------------------------------

def test_dot_lies_on_drawn_line_for_non_normalised_data(tmp_path):
    """The original JS multiplied each y value by the axes height as if
    values were in [0,1]; for ordinary data the dot landed far off the
    line. Now: dot must lie on the rendered line.
    """
    _, _, svg_path = _line_plot(tmp_path, [0, 1, 2, 3, 4],
                                [10, 20, 30, 20, 10])
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")


def test_axes_padding_does_not_shift_dot(tmp_path):
    """Original ``index = round(plotpos/width * (n-1))`` mapping assumed
    data filled the whole axes; with user-set xlim it doesn't.
    """
    fig, ax, _ = _line_plot(tmp_path, [0, 1, 2, 3, 4], [10, 20, 30, 20, 10])
    ax.set_xlim(-10, 14)
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.7, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")


def test_log_y_axis(tmp_path):
    """Log y is the strongest counter-example to the original
    ``cy = bottom - val*(bottom-top)`` linear assumption.
    """
    fig, ax, _ = _line_plot(tmp_path, [1, 2, 3, 4, 5],
                            [1, 10, 100, 1000, 10000])
    ax.set_yscale("log")
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")


def test_log_x_axis(tmp_path):
    """Log x: data x positions are not linearly spaced in screen pixels,
    so the old linear plotpos->index mapping was wrong.
    """
    fig, ax, _ = _line_plot(tmp_path, [1, 10, 100, 1000, 10000],
                            [10, 20, 30, 20, 10])
    ax.set_xscale("log")
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.3, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")


def test_no_legend(tmp_path):
    """Original code did ``d3.select("#legend_1")`` which would error on
    a plot without a legend. Must still work.
    """
    _, _, svg_path = _line_plot(tmp_path, [0, 1, 2, 3, 4],
                                [10, 20, 30, 20, 10], legend=False)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")


def test_multiple_lines_have_independent_dot_positions(tmp_path):
    """Two lines at the same x but different y must each have a dot on
    their OWN line, not on a single shared value.
    """
    fig, ax = plt.subplots()
    line_a, = ax.plot([0, 1, 2, 3, 4], [10, 20, 30, 20, 10], label="a")
    line_b, = ax.plot([0, 1, 2, 3, 4], [50, 40, 35, 40, 50], label="b")
    line_a.set_gid("series_0")
    line_b.set_gid("series_1")
    ax.legend()
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")
        _assert_dot_on_series(page, "dot_1", "series_1")
        cy_a = _attr_float(page, "#dot_0", "cy")
        cy_b = _attr_float(page, "#dot_1", "cy")
        assert abs(cy_a - cy_b) > 5, "dots should be at distinct y"


def test_date_x_axis(tmp_path):
    """matplotlib stores datetimes as float ordinals internally; the JS
    shouldn't need to know that.
    """
    import datetime
    xs = [datetime.date(2024, 1, 1) + datetime.timedelta(days=7 * i)
          for i in range(5)]
    _, _, svg_path = _line_plot(tmp_path, xs, [10, 20, 30, 20, 10])
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")


def test_overlay_hidden_outside_axes(tmp_path):
    """Hovering outside the axes area must hide the overlay rather than
    snap the dot to an arbitrary point.
    """
    _, _, svg_path = _line_plot(tmp_path, [0, 1, 2, 3, 4],
                                [10, 20, 30, 20, 10])
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, -0.5)  # well above the axes
        assert page.locator("#hover_overlay").get_attribute(
            "visibility") == "hidden"


def test_fill_between_dot_on_top_edge(tmp_path):
    """``fill_between`` produces a PolyCollection, not a Line2D. The
    hover should still snap a dot to the top edge of the filled area.
    """
    fig, ax = plt.subplots()
    coll = ax.fill_between([0, 1, 2, 3, 4], [10, 20, 30, 20, 10],
                           alpha=0.4, label="area")
    coll.set_gid("series_0")
    ax.legend()
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")


def test_stackplot_dots_on_each_layer_top(tmp_path):
    """``ax.stackplot`` returns a list of PolyCollections, one per
    layer, each drawn at its cumulative top.
    """
    fig, ax = plt.subplots()
    stacks = ax.stackplot([0, 1, 2, 3, 4],
                          [10, 10, 10, 10, 10],
                          [20, 20, 30, 20, 10], labels=["a", "b"])
    stacks[0].set_gid("series_0")
    stacks[1].set_gid("series_1")
    ax.legend()
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        _assert_dot_on_series(page, "dot_0", "series_0")
        _assert_dot_on_series(page, "dot_1", "series_1")
        cy_a = _attr_float(page, "#dot_0", "cy")
        cy_b = _attr_float(page, "#dot_1", "cy")
        # layer 'a' (cumulative=a) is below 'b' (cumulative=a+b) on
        # screen, so its SVG cy is larger (further down).
        assert cy_a > cy_b, (
            f"layer 'a' should be below 'b': cy_a={cy_a}, cy_b={cy_b}")


def test_bar_dot_on_top_of_bar(tmp_path):
    """``ax.bar`` produces a BarContainer of Rectangle patches. The
    hover should snap a dot to the top-center of the bar at the
    cursor's x position.
    """
    fig, ax = plt.subplots()
    bars = ax.bar([0, 1, 2, 3, 4], [10, 20, 30, 20, 10], label="bars")
    for i, b in enumerate(bars):
        b.set_gid(f"bar_{i}")
    ax.legend()
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        cx = _attr_float(page, "#dot_0", "cx")
        cy = _attr_float(page, "#dot_0", "cy")
        assert cx is not None and cy is not None, "overlay not visible"
        expected_y = page.evaluate("""(cx) => {
            let top = null;
            for (let i = 0; i < 5; i++) {
                const el = document.getElementById('bar_' + i);
                if (!el) continue;
                const bb = el.getBBox();
                if (cx >= bb.x && cx <= bb.x + bb.width
                    && (top === null || bb.y < top)) { top = bb.y; }
            }
            return top;
        }""", cx)
        assert expected_y is not None, (
            f"no bar contains the dot's x={cx}")
        assert cy == pytest.approx(expected_y, abs=1.5), (
            f"dot at ({cx:.1f}, {cy:.1f}) not on bar top "
            f"(expected y={expected_y:.1f})")


def test_dot_hidden_for_series_in_other_axes(tmp_path):
    """In a multi-subplot figure, hovering inside subplot A must not
    move (or show) the dot for a series that belongs to subplot B.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    line_a, = ax1.plot([0, 1, 2, 3, 4], [10, 20, 30, 20, 10], label="a")
    line_b, = ax2.plot([0, 1, 2, 3, 4], [50, 40, 35, 40, 50], label="b")
    line_a.set_gid("series_0")
    line_b.set_gid("series_1")
    ax1.legend()
    ax2.legend()
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5, axes_gid="axes_1")
        _assert_dot_on_series(page, "dot_0", "series_0")
        assert page.locator("#dot_1").get_attribute("visibility") == "hidden"


def test_tooltip_background_matches_legend(tmp_path):
    """The tooltip box should pick up the legend's frame colour so it
    blends with whatever theme matplotlib is rendering.
    """
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1, 2, 3, 4], [10, 20, 30, 20, 10], label="s")
    line.set_gid("series_0")
    legend = ax.legend(facecolor="#ffeecc", edgecolor="#884400",
                       labelcolor="#222222")
    legend.get_frame().set_alpha(1.0)
    svg_path = _save(fig, tmp_path)

    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5)
        rgb = page.evaluate(
            "() => {"
            " const s = getComputedStyle("
            "  document.getElementById('hover_tooltip_box'));"
            " return [s.backgroundColor, s.borderTopColor, s.color];"
            "}")
    # Browsers normalise hex to rgb(...).
    assert rgb[0] == "rgb(255, 238, 204)", f"bg={rgb[0]!r}"
    assert rgb[1] == "rgb(136, 68, 0)", f"border={rgb[1]!r}"
    assert rgb[2] == "rgb(34, 34, 34)", f"text={rgb[2]!r}"


def test_legend_hidden_only_in_active_axes(tmp_path):
    """When hovering inside one subplot, its legend should be hidden so
    the tooltip isn't visually competing with it -- but the OTHER
    subplot's legend should stay visible. Moving the cursor out of the
    figure restores both.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    line_a, = ax1.plot([0, 1, 2, 3, 4], [10, 20, 30, 20, 10], label="a")
    line_b, = ax2.plot([0, 1, 2, 3, 4], [50, 40, 35, 40, 50], label="b")
    line_a.set_gid("series_0")
    line_b.set_gid("series_1")
    leg1 = ax1.legend()
    leg2 = ax2.legend()
    leg1.set_gid("hover_legend_0")
    leg2.set_gid("hover_legend_1")
    svg_path = _save(fig, tmp_path)

    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5, axes_gid="axes_1")
        vis1_hovering = page.locator("#hover_legend_0").get_attribute(
            "visibility")
        vis2_hovering = page.locator("#hover_legend_1").get_attribute(
            "visibility")
        # Move cursor well outside the figure to trigger mouseleave.
        page.mouse.move(0, 0)
        page.wait_for_timeout(100)
        vis1_after = page.locator("#hover_legend_0").get_attribute(
            "visibility")
        vis2_after = page.locator("#hover_legend_1").get_attribute(
            "visibility")
    assert vis1_hovering == "hidden", f"legend_0 vis={vis1_hovering!r}"
    assert vis2_hovering != "hidden", (
        f"legend_1 should remain visible: vis={vis2_hovering!r}")
    assert vis1_after != "hidden", f"legend_0 not restored: vis={vis1_after!r}"
    assert vis2_after != "hidden", f"legend_1 not restored: vis={vis2_after!r}"


def test_tooltip_box_stays_inside_figure(tmp_path):
    """The visible tooltip content must not extend past the figure
    edges, even when hovering near the right edge of an axes.
    """
    _, _, svg_path = _line_plot(tmp_path, [0, 1, 2, 3, 4],
                                [10, 20, 30, 20, 10])
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.95, 0.5)
        fig_bb = page.locator("svg").bounding_box()
        tt_bb = page.locator("#hover_tooltip_box").bounding_box()
    assert tt_bb is not None, "tooltip not rendered"
    # Allow 2 CSS-px slop for sub-pixel rounding.
    assert tt_bb["x"] >= fig_bb["x"] - 2, (
        f"tooltip overflows left: tt.x={tt_bb['x']} fig.x={fig_bb['x']}")
    assert tt_bb["x"] + tt_bb["width"] <= fig_bb["x"] + fig_bb["width"] + 2


def test_tooltip_stays_inside_figure_multi_axes(tmp_path):
    """In a multi-axes figure, hovering in the middle of a narrow
    subplot must still keep the visible tooltip inside the FIGURE
    (not necessarily inside that subplot's axes).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    line_a, = ax1.plot([0, 1, 2, 3, 4], [10, 20, 30, 20, 10], label="aaaa")
    line_a.set_gid("series_0")
    ax1.legend()
    line_b, = ax2.plot([0, 1, 2, 3, 4], [50, 40, 35, 40, 50], label="bbbb")
    line_b.set_gid("series_1")
    ax2.legend()
    svg_path = _save(fig, tmp_path)
    with _open(svg_path) as page:
        _hover_at_axes_fraction(page, 0.5, 0.5, axes_gid="axes_1")
        fig_bb = page.locator("svg").bounding_box()
        tt_bb = page.locator("#hover_tooltip_box").bounding_box()
    assert tt_bb is not None, "tooltip not rendered"
    assert tt_bb["x"] >= fig_bb["x"] - 2, (
        f"tooltip overflows left: tt.x={tt_bb['x']} fig.x={fig_bb['x']}")
    assert tt_bb["x"] + tt_bb["width"] <= fig_bb["x"] + fig_bb["width"] + 2
