"""
Configuration for the order of gallery sections and examples.
Paths are relative to the conf.py file.
"""

from sphinx_gallery.sorting import ExplicitOrder
import os


# Utility functions
def get_ordering(dir, include_directory_path=False):
    """Read gallery_order.txt in dir and return content of the file as a List"""
    file_path = os.path.join(dir, 'gallery_order.txt')
    f = open(file_path, "r")
    lines = [line.replace('\n', '') for line in f.readlines()]
    ordered_list = []
    for line in lines:
        if line == "unsorted":
            ordered_list.append(UNSORTED)
        else:
            if include_directory_path:
                ordered_list.append(os.path.join(dir, line))
            else:
                ordered_list.append(line)

    return ordered_list


def list_directory(parent_dir):
    """Return list of sub directories at a directory"""
    root, dirs, files = next(os.walk(parent_dir))
    return [os.path.join(root, dir) for dir in dirs]

# Gallery sections shall be displayed in the following order.
# Non-matching sections are inserted at the unsorted position

UNSORTED = "unsorted"


plot_types_directory = "../galleries/plot_types/"
plot_types_order = get_ordering(plot_types_directory, include_directory_path=True)

examples_directory = "../galleries/examples/"
examples_order = get_ordering(examples_directory, include_directory_path=True)

folder_lists = [examples_order, plot_types_order]
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
    # folders that don't contain gallery_order.txt file can  
    # list their file orderings here 
]


for dir in list_directory(plot_types_directory):
    try:
        ordered_subdirs = get_ordering(dir, include_directory_path=False)
        list_all.extend(ordered_subdirs)
    except FileNotFoundError:
        # Fallback to ordering already defined in list_all
        pass


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
