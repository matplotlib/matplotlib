"""
Configuration for the order of gallery sections and examples.
Paths are relative to the conf.py file.
"""

from sphinx_gallery.sorting import ExplicitOrder

# Gallery sections shall be displayed in the following order.
# Non-matching sections are appended.
explicit_order_folders = [
                          '../examples/lines_bars_and_markers',
                          '../examples/images_contours_and_fields',
                          '../examples/subplots_axes_and_figures',
                          '../examples/statistics',
                          '../examples/pie_and_polar_charts',
                          '../examples/text_labels_and_annotations',
                          '../examples/pyplots',
                          '../examples/color',
                          '../examples/shapes_and_collections',
                          '../examples/style_sheets',
                          '../examples/axes_grid1',
                          '../examples/axisartist',
                          '../examples/showcase',
                          '../tutorials/introductory',
                          '../tutorials/intermediate',
                          '../tutorials/advanced',
                          '../plot_types/basic',
                          '../plot_types/arrays',
                          '../plot_types/stats',
                          '../plot_types/unstructured',
                          ]


class MplExplicitOrder(ExplicitOrder):
    """For use within the 'subsection_order' key."""
    def __call__(self, item):
        """Return a string determining the sort order."""
        if item in self.ordered_list:
            return "{:04d}".format(self.ordered_list.index(item))
        else:
            # ensure not explicitly listed items come last.
            return "zzz" + item


# Subsection order:
# Subsections are ordered by filename, unless they appear in the following
# lists in which case the list order determines the order within the section.
# Examples/tutorials that do not appear in a list will be appended.

list_all = [
    #  **Tutorials**
    #  introductory
    "usage", "pyplot", "sample_plots", "images", "lifecycle", "customizing",
    #  intermediate
    "artists", "legend_guide", "color_cycle", "gridspec",
    "constrainedlayout_guide", "tight_layout_guide",
    #  advanced
    #  text
    "text_intro", "text_props",
    #  colors
    "colors",

    #  **Examples**
    #  color
    "color_demo",
    #  pies
    "pie_features", "pie_demo2",

    # **Plot Types
    # Basic
    "plot", "scatter_plot", "bar", "stem", "step", "fill_between",
    # Arrays
    "imshow", "pcolormesh", "contour", "contourf",
    "barbs", "quiver", "streamplot",
    # Stats
    "hist_plot", "boxplot_plot", "errorbar_plot", "violin",
    "eventplot", "hist2d", "hexbin", "pie",
    # Unstructured
    "tricontour", "tricontourf", "tripcolor", "triplot",
    ]
explicit_subsection_order = [item + ".py" for item in list_all]


class MplExplicitSubOrder:
    """For use within the 'within_subsection_order' key."""
    def __init__(self, src_dir):
        self.src_dir = src_dir  # src_dir is unused here
        self.ordered_list = explicit_subsection_order

    def __call__(self, item):
        """Return a string determining the sort order."""
        if item in self.ordered_list:
            return "{:04d}".format(self.ordered_list.index(item))
        else:
            # ensure not explicitly listed items come last.
            return "zzz" + item


# Provide the above classes for use in conf.py
sectionorder = MplExplicitOrder(explicit_order_folders)
subsectionorder = MplExplicitSubOrder
