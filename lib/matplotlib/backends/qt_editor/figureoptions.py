# -*- coding: utf-8 -*-
#
# Copyright © 2009 Pierre Raybaut
# Licensed under the terms of the MIT License
# see the mpl licenses directory for a copy of the license


"""Module that provides a GUI-based editor for matplotlib's figure options"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os.path as osp

import matplotlib.backends.qt_editor.formlayout as formlayout
from matplotlib.backends.qt_compat import QtGui
from matplotlib import markers


def get_icon(name):
    import matplotlib
    basedir = osp.join(matplotlib.rcParams['datapath'], 'images')
    return QtGui.QIcon(osp.join(basedir, name))

LINESTYLES = {'-': 'Solid',
              '--': 'Dashed',
              '-.': 'DashDot',
              ':': 'Dotted',
              'steps': 'Steps',
              'none': 'None',
              }

MARKERS = markers.MarkerStyle.markers


def figure_edit(axes, parent=None):
    """Edit matplotlib figure options"""
    sep = (None, None)  # separator

    has_curve = len(axes.get_lines()) > 0

    # Get / General
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
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

    if has_curve:
        # Get / Curves
        linedict = {}
        for line in axes.get_lines():
            label = line.get_label()
            if label == '_nolegend_':
                continue
            linedict[label] = line
        curves = []
        linestyles = list(six.iteritems(LINESTYLES))
        markers = list(six.iteritems(MARKERS))
        curvelabels = sorted(linedict.keys())
        for label in curvelabels:
            line = linedict[label]
            curvedata = [('Label', label),
                         sep,
                         (None, '<b>Line</b>'),
                         ('Style', [line.get_linestyle()] + linestyles),
                         ('Width', line.get_linewidth()),
                         ('Color', line.get_color()),
                         sep,
                         (None, '<b>Marker</b>'),
                         ('Style', [line.get_marker()] + markers),
                         ('Size', line.get_markersize()),
                         ('Facecolor', line.get_markerfacecolor()),
                         ('Edgecolor', line.get_markeredgecolor()),
                         ]
            curves.append([curvedata, label, ""])

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

        if has_curve:
            # Set / Curves
            for index, curve in enumerate(curves):
                line = linedict[curvelabels[index]]
                label, linestyle, linewidth, color, \
                    marker, markersize, markerfacecolor, markeredgecolor \
                    = curve
                line.set_label(label)
                line.set_linestyle(linestyle)
                line.set_linewidth(linewidth)
                line.set_color(color)
                if marker is not 'none':
                    line.set_marker(marker)
                    line.set_markersize(markersize)
                    line.set_markerfacecolor(markerfacecolor)
                    line.set_markeredgecolor(markeredgecolor)

        # re-generate legend, if checkbox is checked
        if generate_legend:
            if axes.legend_ is not None:
                old_legend = axes.get_legend()
                new_legend = axes.legend(ncol=old_legend._ncol)
                new_legend.draggable(old_legend._draggable is not None)
            else:
                new_legend = axes.legend()
                new_legend.draggable(True)

        # Redraw
        figure = axes.get_figure()
        figure.canvas.draw()

    data = formlayout.fedit(datalist, title="Figure options", parent=parent,
                            icon=get_icon('qt4_editor_options.svg'),
                            apply=apply_callback)
    if data is not None:
        apply_callback(data)
