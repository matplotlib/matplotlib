"""
Configuration for the order of gallery sections and examples.
Paths are relative to the conf.py file.
"""

import itertools
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder
from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)

# Gallery sections shall be displayed in the following order.
# Non-matching sections are inserted at the unsorted position

UNSORTED = "unsorted"

examples_order = [
    # plotting data
    '../galleries/examples/lines_bars_and_markers',
    '../galleries/examples/images_contours_and_fields',
    '../galleries/examples/statistics',
    '../galleries/examples/pie_and_polar_charts',
    # figure, Axes, subplots
    '../galleries/examples/subplots_axes_and_figures',
    # axis properties
    '../galleries/examples/ticks',
    '../galleries/examples/scales',
    '../galleries/examples/spines',
    # decorations, individual artists
    '../galleries/examples/text_labels_and_annotations',
    '../galleries/examples/shapes_and_collections',
    # styling
    '../galleries/examples/color',
    '../galleries/examples/style_sheets',
    # interfaces / add-ons
    '../galleries/examples/pyplots',
    '../galleries/examples/mplot3d',
    '../galleries/examples/axes_grid1',
    '../galleries/examples/axisartist',
    # animation, interactivity
    '../galleries/examples/animation',
    '../galleries/examples/widgets',
    '../galleries/examples/event_handling',
    '../galleries/examples/user_interfaces',
    UNSORTED,
    '../galleries/examples/misc',
    # nice and special visualizations
    '../galleries/examples/showcase',
    '../galleries/examples/specialty_plots',
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


class MplFileExplicitOrder(ExplicitOrder):
    """
    An explicit order class that reads the order of examples from 'gallery_order.txt'.

    For use with the sphinx_gallery 'within_subsection_order' key.

    The file contains a list of example filenames (without the .py extension) in the
    desired order, with an optional '*' to indicate where not-listed examples should be
    placed.

    If '*' is not present, all examples must be listed, or an error will be raised.
    Use this if you want to ensure that a full order is intentionally maintained.
    """
    def __init__(self, src_dir):
        ordered_list = self.read_gallery_order(Path(src_dir).resolve()) or []
        super().__init__(ordered_list)

    @staticmethod
    def read_gallery_order(src_dir: Path):
        """Return the list of examples to be sorted; read from 'gallery_order.txt'."""
        gallery_order_txt = src_dir / "gallery_order.txt"
        if not gallery_order_txt.exists():
            return None
        lines = [
            line.strip()
            for line in gallery_order_txt.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

        try:
            placeholder_index = lines.index("*")
        except ValueError:
            placeholder_index = None

        lines = [line + ".py" for line in lines]

        if placeholder_index is None:
            front = lines
            back = []
        else:
            front = lines[:placeholder_index]
            back = lines[placeholder_index+1:]

        listed_examples = {*front, *back}
        existing_examples = {
            file.name for file in src_dir.iterdir() if file.suffix == ".py"
        }

        non_existing_examples = listed_examples - existing_examples
        missing_examples = existing_examples - listed_examples

        rel_txt_path = gallery_order_txt.relative_to(gallery_order_txt.parents[3])
        if non_existing_examples:
            logger.warning(
                f"The following examples listed in {rel_txt_path} do not exist: "
                f"{', '.join(non_existing_examples)}")
        if placeholder_index is None and missing_examples:
            logger.warning(
                f"The following examples are not listed in {rel_txt_path}. "
                f"Either include them or add a '*' to indicate where not listed "
                f"examples should be placed: "
                f"{', '.join(missing_examples)}"
            )

        mid = list(sorted(missing_examples))
        return front + mid + back

    def __call__(self, item):
        """Return a string determining the sort order."""
        if not self.ordered_list:
            return item
        return f"{self.ordered_list.index(item):04d}"

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.ordered_list)

# Provide the above classes for use in conf.py
sectionorder = MplExplicitOrder(explicit_order_folders)
subsectionorder = MplFileExplicitOrder

_preserve_count = itertools.count()


def preserve_order(item):
    """A sorting key to preserve the original order of items in minigalleries."""
    return next(_preserve_count)
