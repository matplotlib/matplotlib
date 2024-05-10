"""
Configuration for the order of gallery sections and examples.
Paths are relative to the conf.py file.
"""

from sphinx_gallery.sorting import ExplicitOrder

# Gallery sections shall be displayed in the following order.
# Non-matching sections are inserted at the unsorted position

UNSORTED = "unsorted"

examples_order = [
    '../galleries/examples/lines_bars_and_markers',
    '../galleries/examples/images_contours_and_fields',
    '../galleries/examples/subplots_axes_and_figures',
    '../galleries/examples/statistics',
    '../galleries/examples/pie_and_polar_charts',
    '../galleries/examples/text_labels_and_annotations',
    '../galleries/examples/color',
    '../galleries/examples/shapes_and_collections',
    '../galleries/examples/style_sheets',
    '../galleries/examples/pyplots',
    '../galleries/examples/axes_grid1',
    '../galleries/examples/axisartist',
    '../galleries/examples/showcase',
    UNSORTED,
    '../galleries/examples/userdemo',
]

tutorials_order = [
    '../galleries/tutorials/introductory',
    '../galleries/tutorials/intermediate',
    '../galleries/tutorials/advanced',
    UNSORTED,
    '../galleries/tutorials/provisional'
]

plot_types_order = [
    '../galleries/plot_types/basic',
    '../galleries/plot_types/stats',
    '../galleries/plot_types/arrays',
    '../galleries/plot_types/unstructured',
    '../galleries/plot_types/3D',
    UNSORTED
]

folder_lists = [examples_order, tutorials_order, plot_types_order]

explicit_order_folders = [fd for folders in folder_lists
                          for fd in folders[:folders.index(UNSORTED)]]
explicit_order_folders.append(UNSORTED)
explicit_order_folders.extend([fd for folders in folder_lists
                               for fd in folders[folders.index(UNSORTED):]])


class MplExplicitOrder(ExplicitOrder):
    """For use within the 'subsection_order' key."""
    def __call__(self, item):
        """Return a string determining the sort order."""
        if item in self.ordered_list:
            return f"{self.ordered_list.index(item):04d}"
        else:
            return f"{self.ordered_list.index(UNSORTED):04d}{item}"

# Subsection order:
# Subsections are ordered by filename, unless they appear in the following
# lists in which case the list order determines the order within the section.
# Examples/tutorials that do not appear in a list will be appended.

list_all = [
    #  **Tutorials**
    #  introductory
    "quick_start", "pyplot", "images", "lifecycle", "customizing",
    #  intermediate
    "artists", "legend_guide", "color_cycle",
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
    # Spines
    "spines", "spine_placement_demo", "spines_dropped",
    "multiple_yaxis_with_spines", "centered_spines_with_arrows",
    ]
explicit_subsection_order = [item + ".py" for item in list_all]


class MplExplicitSubOrder(ExplicitOrder):
    """For use within the 'within_subsection_order' key."""
    def __init__(self, src_dir):
        self.src_dir = src_dir  # src_dir is unused here
        self.ordered_list = explicit_subsection_order

    def __call__(self, item):
        """Return a string determining the sort order."""
        if item in self.ordered_list:
            return f"{self.ordered_list.index(item):04d}"
        else:
            # ensure not explicitly listed items come last.
            return "zzz" + item


# Provide the above classes for use in conf.py
sectionorder = MplExplicitOrder(explicit_order_folders)
subsectionorder = MplExplicitSubOrder
