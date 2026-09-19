#!/usr/bin/env python
"""
Generates layered SVG files used for the macOS backend app icons.
"""

# ------------------------------------------------------------------------------------
#
# Usage:
#
# python tools/make_macos_svgs.py /path/to/tmp/folder
# python tools/make_icons.py --macos --source-dir=/path/to/tmp/folder
#
# ------------------------------------------------------------------------------------
#
# Notes:
#
# The goal of this file is to allow maintainers to modify/tweak the macOS app icons
# without having to use an image editor or macOS-native tool.
#
# ------------------------------------------------------------------------------------
#
# Implementation Details:
#
# Our approach to macOS-native icons involves compositing several PNG files:
#
# macos_appicon_shadow*.png - Icon shadow
# macos_appicon_bgdark*.png - Background gradient and bezel, dark appearance
# macos_appicon_mask*.png   - Squircle mask
# macos_appicon_light.png   - Actual icon contents, light appearance
# macos_appicon_dark.png    - Actual icon contents, dark appearance
#
# None of this should be necessary. Apple should provide API to programmatically
# generate a native app icon from a template image. This would allow us to simply
# have the 'macos_appicon_light' and 'macos_appicon_dark' content files.
#
# In an ideal future, there would be a SVG renderer that supports modern
# CSS features like color-mask() and CSS Variables. We could then generate these
# two files from a shared 'macos_appicon_template.svg'. This would be similar to
# the approach used for other files in `mpl-data/images'.
#
# In the meantime, we have to construct all of the SVG files in this script.
#
# SquircleGenerator creates the 'shadow' and 'mask' SVG files.
# AppIconGenerator creates the 'light' and 'dark' SVG files.
# The SVG files are then rendered to PNGs using the 'make_icons.py' tool.
#
# The 'bgdark' files are manually-generated using a native tool and should be viewed
# as a nicety rather than a necessity. If not provided, the macOS backend will draw a
# simple gradient.
#
# ------------------------------------------------------------------------------------

import argparse
import os
from pathlib import Path
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html import escape
from enum import IntEnum


DEFAULT_DEST_PATH = "/tmp/matplotlib_macos_svgs"


class Appearance(IntEnum):
    LIGHT = 1
    DARK = 2


GRID_CIRCLE_RADII = [36, 60, 84]


@dataclass(slots=True)
class AppIconSector:
    color: str
    stroke_width: float
    radius: float
    color_adjustments: dict[Appearance, dict]
    path: str


@dataclass(slots=True)
class AppIconVariables:
    shadow_opacity: float
    background_grid_color: str
    background_grid_opacity: float
    sector_grid_opacity: float
    background_top_color: str | None
    background_bottom_color: str | None
    middle_dot_color: str


APPICON_VARS = {
    Appearance.LIGHT: AppIconVariables(
        shadow_opacity=0.2,
        background_grid_color="#000000",
        background_grid_opacity=0.15,
        sector_grid_opacity=0.1,
        background_top_color="#fcfcfc",
        background_bottom_color="#f8f8f8",
        middle_dot_color="#303030"
    ),
    Appearance.DARK: AppIconVariables(
        shadow_opacity=1.0,
        background_grid_color="#ffffff",
        background_grid_opacity=0.4,
        sector_grid_opacity=0.5,
        background_top_color=None,
        background_bottom_color=None,
        middle_dot_color="#dddddd"
    )
}


APPICON_SECTORS = (
    # Sector 0 - jet(0.6) green
    AppIconSector(
        color="#ceff29", stroke_width=1.5, radius=72,
        color_adjustments={
            Appearance.LIGHT: {"fill": (0.4, 0.3), "stroke": -0.6},
            Appearance.DARK:  {"fill": (0.2, 0.4), "stroke":  0.5}
        },
        path=(
            "m128 128 24.7 -65.815q0.625 -1.65 2.28 -1.035 4.425 1.815 8.585 4.17"
            "a76 76 0 0 1 7.985 5.235q1.38 1.1 0.28 2.485z"
        )
    ),

    # Sector 1 - jet(0.2) blue
    AppIconSector(
        color="#004cff", stroke_width=1.5, radius=24,
        color_adjustments={
            Appearance.LIGHT: {"fill": (0.4, 0.2), "stroke": -0.4},
            Appearance.DARK:  {"fill": (0.2, 0.4), "stroke":  0.5}
        },
        path=(
            "m128 128 22.005 -7.15q0.84 -0.275 1.115 0.57 0.375 1.395 0.605 2.82"
            "q0.225 1.43 0.3 2.875 0 0.885 -0.885 0.885z"
        )
    ),

    # Sector 2 - jet(0.8) dark orange
    AppIconSector(
        color="#ff6800", stroke_width=1.5, radius=96,
        color_adjustments={
            Appearance.LIGHT: {"fill": (0.4, 0.2), "stroke": -0.5},
            Appearance.DARK:  {"fill": (0.2, 0.4), "stroke":  0.5}
        },
        path=(
            "m128 128 73.7 57.435q2.095 1.625 0.46 3.72 -2.79 3.315 -5.835 6.41"
            "a112 112 0 0 1 -6.34 5.905q-2.075 1.65 -3.73 -0.42z"
        )
    ),

    # Sector 3 - jet(0.5) mint
    AppIconSector(
        color="#7dff7a", stroke_width=1.5, radius=60,
        color_adjustments={
            Appearance.LIGHT: {"fill": (0.4, 0.2), "stroke": -0.5},
            Appearance.DARK:  {"fill": (0.2, 0.4), "stroke":  0.5}
        },
        path=(
            "m128 128 9.765 57.465q0.3 1.74 -1.45 2.04 -4.98 0.66 -10 0.525"
            "q-5.02 -0.145 -9.95 -1.085 -1.73 -0.4 -1.335 -2.12z"
        )
    ),

    # Sector 4 - jet(0.4) cyan
    AppIconSector(
        color="#29ffce", stroke_width=1.5, radius=48,
        color_adjustments={
            Appearance.LIGHT: {"fill": (0.4, 0.2), "stroke": -0.6},
            Appearance.DARK:  {"fill": (0.2, 0.4), "stroke":  0.5}
        },
        path=(
            "m128 128 -38.765 26.855q-0.73 0.51 -1.23 -0.22a56 56 0 0 1 -2.91 -4.99"
            "q-0.38 -0.8 0.415 -1.185z"
        )
    ),

    # Sector 5 - jet(0.7) light orange
    AppIconSector(
        color="#ffc400", stroke_width=1.5, radius=84,
        color_adjustments={
            Appearance.LIGHT: {"fill": (0.4, 0.2), "stroke": -0.5},
            Appearance.DARK:  {"fill": (0.2, 0.4), "stroke":  0.5}
        },
        path=(
            "M128 128 47.885 142.54q-2.61 0.47 -3.085 -2.14"
            "a85.5 85.5 0 0 1 -0.115 -23.685 85.5 85.5 0 0 1 6.41 -22.8"
            "q1.15 -2.4 3.545 -1.245z"
        )
    ),

    # Sector 6 - jet(0.8) dark orange
    AppIconSector(
        color="#ff6800", stroke_width=1.5, radius=96,
        color_adjustments={
            Appearance.LIGHT: {"fill": (0.4, 0.2), "stroke": -0.5},
            Appearance.DARK:  {"fill": (0.2, 0.4), "stroke":  0.5}

        },
        path=(
            "M128 128 68.12 56.275q-1.7 -2.035 0.335 -3.74"
            "a98.5 98.5 0 0 1 16.885 -10.635 98.5 98.5 0 0 1 18.69 -6.99"
            "q2.585 -0.59 3.18 2z"
        )
    )
)


def svg_element(
    name: str,
    attrs: Mapping[str, str] | None = None,
    children: Iterable[str] | None = None
) -> str:
    attr_str = (
        " ".join(
            f'{k}="{escape(str(v), quote=True)}"' for k, v in attrs.items()
        ) if attrs else ""
    )

    lines = []
    if children is not None:
        lines.append(f"<{name} {attr_str}>")
        lines.extend(children)
        lines.append(f"</{name}>")
    else:
        lines.append(f"<{name} {attr_str}/>")

    return "\n".join(lines)


def svg_doc(width: int, height: int, children: Iterable[str]) -> str:
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        svg_element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{width}px", "height": f"{height}px",
            "viewBox": f"0 0 {width} {height}"
        }, children)
    ])


def svg_radial_gradient(
    id: str,
    cx: float, cy: float,
    r: float,
    children: Iterable[str]
) -> str:
    return svg_element("radialGradient", {
        "id": id, "gradientUnits": "userSpaceOnUse",
        "cx": str(cx), "cy": str(cy), "r": str(r)
    }, children)


def svg_stop(offset: float, color: str, opacity: float = 1.0) -> str:
    attrs = {"offset": f"{offset*100:.1f}%", "stop-color": color}
    if opacity != 1.0:
        attrs["stop-opacity"] = f"{opacity:.3f}"
    return svg_element("stop", attrs)


def adjust_color(hex: str, adjust: float):
    hex = hex.lstrip("#")

    if len(hex) != 6:
        raise ValueError("Invalid hex color string")

    r = int(hex[0:2], 16)
    g = int(hex[2:4], 16)
    b = int(hex[4:6], 16)

    if adjust > 0:
        r = round(r * (1 - adjust) + 255 * adjust)
        g = round(g * (1 - adjust) + 255 * adjust)
        b = round(b * (1 - adjust) + 255 * adjust)
    elif adjust < 0:
        alpha = abs(adjust)
        r = round(r * (1 - alpha))
        g = round(g * (1 - alpha))
        b = round(b * (1 - alpha))

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass(slots=True)
class SquircleGenerator:
    icon_width: int
    square_width: int
    corner_radius: float

    def continuous_rounded_rect_path_data(self, x: float, y: float) -> str:
        """Generate SVG path data using macOS/iOS squircle formula."""

        # See comments in MPLUtils.m about these constants
        mA = 1.528665
        mB = 1.08849
        mC = 0.868407
        mD = 0.631494
        mE = 0.0749114
        mF = 0.372824
        mG = 0.169060

        commands = []

        def point(x: float, y: float) -> str:
            return f"{x:.6f},{y:.6f}"

        def line(x: float, y: float) -> None:
            commands.append(f"L {point(x, y)}")

        def curve(
            c1x: float, c1y: float,
            c2x: float, c2y: float,
            x: float, y: float
        ) -> None:
            commands.append(f"C {point(c1x, c1y)} {point(c2x, c2y)} {point(x, y)}")

        def corner(
            cx: float, cy: float,
            dx0: float, dy0: float,
            dx1: float, dy1: float
        ) -> None:
            line(cx + (dx0 * mA), cy + (dy0 * mA))

            curve(cx + (dx0 * mB),            cy + (dy0 * mB),
                  cx + (dx0 * mC),            cy + (dy0 * mC),
                  cx + (dx0 * mD + dx1 * mE), cy + (dy0 * mD + dy1 * mE))

            curve(cx + (dx0 * mF + dx1 * mG), cy + (dy0 * mF + dy1 * mG),
                  cx + (dx0 * mG + dx1 * mF), cy + (dy0 * mG + dy1 * mF),
                  cx + (dx0 * mE + dx1 * mD), cy + (dy0 * mE + dy1 * mD))

            curve(cx + (dx1 * mC), cy + (dy1 * mC),
                  cx + (dx1 * mB), cy + (dy1 * mB),
                  cx + (dx1 * mA), cy + (dy1 * mA))

        w = self.square_width
        r = self.corner_radius

        commands.append(f"M {point(x + mA * r, y)}")

        corner(x + w, y, -r,  0,  0,  r)
        corner(x + w, y + w,  0, -r, -r,  0)
        corner(x,     y + w,  r,  0,  0, -r)
        corner(x,     y,      0,  r,  r,  0)

        commands.append("Z")

        return " ".join(commands)

    def make_mask(self) -> str:
        return svg_doc(self.square_width, self.square_width, (
            svg_element("rect", {
                "x": "0", "y": "0", "width": "100%", "height": "100%", "fill": "black"
            }),
            svg_element("path", {
                "d": self.continuous_rounded_rect_path_data(x=0, y=0), "fill": "white"
            })
        ))

    def make_shadow(self, opacity: float, offset: float, radius: float) -> str:
        x = (self.icon_width - self.square_width) / 2
        y = x + offset

        return svg_doc(self.icon_width, self.icon_width, (
            svg_element("defs", children=[
                svg_element("filter", {"id": "blur"}, [
                    svg_element("feGaussianBlur", {"stdDeviation": f"{radius}"})
                ])
            ]),
            svg_element("path", {
                "d": self.continuous_rounded_rect_path_data(x=x, y=y),
                "fill": "black", "fill-opacity": f"{opacity:.3f}",
                "filter": "url(#blur)",
            })
        ))


class AppIconGenerator():

    def __init__(self, appearance: Appearance) -> None:
        self._appearance = appearance
        self._vars = APPICON_VARS[appearance]
        self._defs: list[str] = []
        self._children: list[str] = []

    def add_def(self, child: str) -> None:
        self._defs.extend(("", child))

    def add_child(self, child: str) -> None:
        self._children.extend(("", child))

    def _get_sector_stroke_color(self, sector: AppIconSector) -> str:
        stroke_adjustment = sector.color_adjustments[self._appearance]["stroke"]
        return adjust_color(sector.color, stroke_adjustment)

    def _add_radial_gradient(self, id: str, r: float, children: Iterable[str]) -> None:
        attrs = {
            "id": id, "gradientUnits": "userSpaceOnUse",
            "cx": "128", "cy": "128", "r": f"{r}"
        }
        self.add_def(svg_element("radialGradient", attrs, children))

    def _add_sector_path_defs(self) -> None:
        path_defs = []
        clip_path_defs = []

        for i, sector in enumerate(APPICON_SECTORS):
            path_id = f"sector{i}-path"
            clip_path_id = f"sector{i}-clip-path"

            path_defs.append(svg_element("path", {"id": path_id, "d": sector.path}))

            clip_path_defs.append(svg_element("clipPath", {"id": clip_path_id}, [
                svg_element("use", {"href": f"#{path_id}"})
            ]))

        self._defs.extend(("", *path_defs, *clip_path_defs))

    def _add_grid_defs(self) -> None:
        def path(d: str) -> str:
            return svg_element("path", {
                "d": d, "stroke-width": "2"
            })

        def circle(r: str) -> str:
            return svg_element("circle", {
                "cx": "128", "cy": "128", "r": r,
                "stroke-width": "1.5"
            })

        self.add_def(svg_element("g", {"id": "grid", "fill": "none"}, [
            path("M0 128 256 128"),
            path("M128 0 128 256"),
            path("M0 0 256 256"),
            path("M0 256 256 0"),
            circle(str(GRID_CIRCLE_RADII[0])),
            circle(str(GRID_CIRCLE_RADII[1])),
            circle(str(GRID_CIRCLE_RADII[2]))
        ]))

    def _add_background(self) -> None:
        top_color = self._vars.background_top_color
        bottom_color = self._vars.background_bottom_color
        if top_color is None or bottom_color is None:
            return

        gradient_id = "background-gradient"

        self.add_def(svg_element("linearGradient", {
            "id": "background-gradient",
            # y1/y2 calculated to align with a 208px squircle
            "x1": "50%", "x2": "50%", "y1": "9.375%", "y2": "90.625%"
        }, [svg_stop(0, top_color), svg_stop(1, bottom_color)]))

        self.add_child(svg_element("rect", {
            "x": "0", "y": "0", "width": "100%", "height": "100%",
            "fill": f"url(#{gradient_id})"
        }))

    def _add_sector_shadow(self):
        self._defs.append(svg_element("filter", {"id": "sector-shadow-blur"}, [
            svg_element("feGaussianBlur", {"stdDeviation": "4"})
        ]))

        self.add_child(svg_element("g", {
            "fill": "black",
            "fill-opacity": str(self._vars.shadow_opacity),
            "transform": "translate(0 2)",
            "stroke": "none",
            "filter": "url(#sector-shadow-blur)"
        }, [
            svg_element("use", {"href": f"#sector{i}-path"})
            for i, sector in enumerate(APPICON_SECTORS)
        ]))

    def _add_background_grid(self) -> None:
        gradient_id = "background-grid-gradient"
        color = self._vars.background_grid_color
        opacity = self._vars.background_grid_opacity

        gradient_radius = 125
        offsets = [x / gradient_radius for x in GRID_CIRCLE_RADII]

        self._add_radial_gradient(gradient_id, r=gradient_radius, children=[
            svg_stop(0,          color, opacity * 1.00),
            svg_stop(offsets[0], color, opacity * 0.80),
            svg_stop(offsets[1], color, opacity * 0.65),
            svg_stop(offsets[2], color, opacity * 0.50),
            svg_stop(1,          color, opacity * 0.15)
        ])

        self.add_child(svg_element("use", {
            "href": "#grid",
            "stroke": f"url(#{gradient_id})"
        }))

    def _add_sector_fill_group(self) -> None:
        paths = []

        for i, sector in enumerate(APPICON_SECTORS):
            adjustments = sector.color_adjustments[self._appearance]
            fill_id = f"sector{i}-fill"

            start_fill, end_fill = (
                adjust_color(sector.color, x) for x in adjustments["fill"]
            )

            self._add_radial_gradient(fill_id, sector.radius, (
                svg_stop(0, start_fill),
                svg_stop(1, end_fill)
            ))

            paths.append(svg_element("use", {
                "href": f"#sector{i}-path",
                "fill": f"url(#{fill_id})",
            }))

        self.add_child(svg_element("g", {"fill": "none", "stroke": "none"}, paths))

    def _add_sector_grid(self) -> None:
        for i, sector in enumerate(APPICON_SECTORS):
            self.add_child(svg_element("use", {
                "href": "#grid",
                "stroke": self._get_sector_stroke_color(sector),
                "stroke-opacity": str(self._vars.sector_grid_opacity),
                "clip-path": f"url(#sector{i}-clip-path)"
            }))

    def _add_sector_stroke_group(self) -> None:
        paths = []

        for i, sector in enumerate(APPICON_SECTORS):
            paths.append(svg_element("use", {
                "href": f"#sector{i}-path",
                "stroke": self._get_sector_stroke_color(sector),
                "stroke-width": str(sector.stroke_width)
            }))

        self.add_child(svg_element("g", {"fill": "none", "stroke": "none"}, paths))

    def _add_middle_dot(self) -> None:
        gradient_id = "middle-dot-gradient"
        color = self._vars.middle_dot_color
        middle_dot_gradient = svg_element("radialGradient", {"id": gradient_id}, [
            svg_stop(0.00, color, 1.0),
            svg_stop(0.33, color, 1.0),
            svg_stop(1.00, color, 0.0)
        ])

        middle_dot = svg_element("circle", {
            "cx": "128", "cy": "128", "r": "3",
            "fill": f"url(#{gradient_id})"
        })

        self.add_def(middle_dot_gradient)
        self.add_child(middle_dot)

    def make(self) -> str:
        self._add_background()
        self._add_sector_path_defs()
        self._add_grid_defs()
        self._add_background_grid()
        self._add_sector_shadow()
        self._add_sector_fill_group()
        self._add_sector_grid()
        self._add_sector_stroke_group()
        self._add_middle_dot()

        return svg_doc(256, 256, (
            svg_element("defs", children=self._defs),
            "\n".join(self._children)
        ))


def make_macos_svg_files() -> None:
    parser = argparse.ArgumentParser(
        description="Create the layered SVG files the macOS app icons")

    parser.add_argument(
        "-d", "--dest-dir",
        type=Path,
        default=Path(__file__).parent / DEFAULT_DEST_PATH,
        help="Directory where to write the PNG files.")

    args = parser.parse_args()

    dest_dir = args.dest_dir
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    def write(filename: str, doc: str) -> None:
        with open(dest_dir / filename, "w", encoding="utf-8") as f:
            f.write(doc)

    write("macos_appicon_light.svg", AppIconGenerator(Appearance.LIGHT).make())
    write("macos_appicon_dark.svg",  AppIconGenerator(Appearance.DARK).make())

    gen = SquircleGenerator(256, 206, 45.5)
    write("macos_appicon_mask11.svg", gen.make_mask())
    write("macos_appicon_shadow11.svg",
          gen.make_shadow(opacity=0.3, offset=3, radius=3))

    gen = SquircleGenerator(256, 208, 54.5)
    write("macos_appicon_mask26.svg", gen.make_mask())
    write("macos_appicon_shadow26.svg",
          gen.make_shadow(opacity=0.25, offset=3, radius=4.8))


if __name__ == "__main__":
    make_macos_svg_files()
