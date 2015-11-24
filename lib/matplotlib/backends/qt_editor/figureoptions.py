# -*- coding: utf-8 -*-
#
# Copyright © 2009 Pierre Raybaut
# Licensed under the terms of the MIT License
# see the mpl licenses directory for a copy of the license


"""Module that provides a GUI-based editor for matplotlib's figure options"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import os.path as osp

import matplotlib.backends.qt_editor.formlayout as formlayout
from matplotlib.backends.qt_compat import QtGui
from matplotlib import markers
from matplotlib.colors import colorConverter, rgb2hex


def get_icon(name):
    import matplotlib
    basedir = osp.join(matplotlib.rcParams['datapath'], 'images')
    return QtGui.QIcon(osp.join(basedir, name))

LINESTYLES = {'-': 'Solid',
              '--': 'Dashed',
              '-.': 'DashDot',
              ':': 'Dotted',
              'none': 'None',
              }

DRAWSTYLES = {'default': 'Default',
              'steps': 'Steps',
              }

MARKERS = markers.MarkerStyle.markers


def figure_edit(axes, parent=None):
    """Edit matplotlib figure options"""
    sep = (None, None)  # separator

    has_curve = len(axes.get_lines()) > 0

    # Get / General
    xmin, xmax = map(float, axes.get_xlim())
    ymin, ymax = map(float, axes.get_ylim())
    general = [('Title', axes.get_title()),
               sep,
               (None, "<b>X-Axis</b>"),
               ('Min', xmin), ('Max', xmax),
               ('Label', axes.get_xlabel()),
               ('Scale', [axes.get_xscale(), 'linear', 'log']),
               sep,
               (None, "<b>Y-Axis</b>"),
               ('Min', ymin), ('Max', ymax),
               ('Label', axes.get_ylabel()),
               ('Scale', [axes.get_yscale(), 'linear', 'log']),
               sep,
               ('(Re-)Generate automatic legend', False),
               ]

    # Save the unit data
    xconverter = axes.xaxis.converter
    yconverter = axes.yaxis.converter
    xunits = axes.xaxis.get_units()
    yunits = axes.yaxis.get_units()

    if has_curve:
        # Get / Curves
        linedict = {}
        for line in axes.get_lines():
            label = line.get_label()
            if label == '_nolegend_':
                continue
            linedict[label] = line
        curves = []

        def prepare_data(d, init):
            """Prepare entry for FormLayout.

            `d` is a mapping of shorthands to style names (a single style may
            have multiple shorthands, in particular the shorthands `None`,
            `"None"`, `"none"` and `""` are synonyms); `init` is one shorthand
            of the initial style.

            This function returns an list suitable for initializing a
            FormLayout combobox, namely `[initial_name, (shorthand,
            style_name), (shorthand, style_name), ...]`.
            """
            # Drop duplicate shorthands from dict (by overwriting them during
            # the dict comprehension).
            name2short = {name: short for short, name in d.items()}
            # Convert back to {shorthand: name}.
            short2name = {short: name for name, short in name2short.items()}
            # Find the kept shorthand for the style specified by init.
            canonical_init = name2short[d[init]]
            # Sort by representation and prepend the initial value.
            return ([canonical_init] +
                    sorted(short2name.items(),
                           key=lambda short_and_name: short_and_name[1]))

        curvelabels = sorted(linedict.keys())
        for label in curvelabels:
            line = linedict[label]
            color = rgb2hex(colorConverter.to_rgb(line.get_color()))
            ec = rgb2hex(colorConverter.to_rgb(line.get_markeredgecolor()))
            fc = rgb2hex(colorConverter.to_rgb(line.get_markerfacecolor()))
            curvedata = [
                ('Label', label),
                sep,
                (None, '<b>Line</b>'),
                ('Line Style', prepare_data(LINESTYLES, line.get_linestyle())),
                ('Draw Style', prepare_data(DRAWSTYLES, line.get_drawstyle())),
                ('Width', line.get_linewidth()),
                ('Color', color),
                sep,
                (None, '<b>Marker</b>'),
                ('Style', prepare_data(MARKERS, line.get_marker())),
                ('Size', line.get_markersize()),
                ('Facecolor', fc),
                ('Edgecolor', ec)]
            curves.append([curvedata, label, ""])

        # make sure that there is at least one displayed curve
        has_curve = bool(curves)

    datalist = [(general, "Axes", "")]
    if has_curve:
        datalist.append((curves, "Curves", ""))

    def apply_callback(data):
        """This function will be called to apply changes"""
        if has_curve:
            general, curves = data
        else:
            general, = data

        # Set / General
        title, xmin, xmax, xlabel, xscale, ymin, ymax, ylabel, yscale, \
            generate_legend = general
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_title(title)
        axes.set_xlim(xmin, xmax)
        axes.set_xlabel(xlabel)
        axes.set_ylim(ymin, ymax)
        axes.set_ylabel(ylabel)

        # Restore the unit data
        axes.xaxis.converter = xconverter
        axes.yaxis.converter = yconverter
        axes.xaxis.set_units(xunits)
        axes.yaxis.set_units(yunits)
        axes.xaxis._update_axisinfo()
        axes.yaxis._update_axisinfo()

        if has_curve:
            # Set / Curves
            for index, curve in enumerate(curves):
                line = linedict[curvelabels[index]]
                label, linestyle, drawstyle, linewidth, color, \
                    marker, markersize, markerfacecolor, markeredgecolor \
                    = curve
                line.set_label(label)
                line.set_linestyle(linestyle)
                line.set_drawstyle(drawstyle)
                line.set_linewidth(linewidth)
                line.set_color(color)
                if marker is not 'none':
                    line.set_marker(marker)
                    line.set_markersize(markersize)
                    line.set_markerfacecolor(markerfacecolor)
                    line.set_markeredgecolor(markeredgecolor)

        # re-generate legend, if checkbox is checked

        if generate_legend:
            draggable = None
            ncol = None
            if axes.legend_ is not None:
                old_legend = axes.get_legend()
                draggable = old_legend._draggable is not None
                ncol = old_legend._ncol
            new_legend = axes.legend(ncol=ncol)
            if new_legend:
                new_legend.draggable(draggable)

        # Redraw
        figure = axes.get_figure()
        figure.canvas.draw()

    data = formlayout.fedit(datalist, title="Figure options", parent=parent,
                            icon=get_icon('qt4_editor_options.svg'),
                            apply=apply_callback)
    if data is not None:
        apply_callback(data)
