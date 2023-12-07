"""Preprocessor for merging consecutive stream outputs for easier handling."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import functools
import re

from traitlets.log import get_logger


def cell_preprocessor(function):
    """
    Wrap a function to be executed on all cells of a notebook

    The wrapped function should have these parameters:

    cell : NotebookNode cell
        Notebook cell being processed
    resources : dictionary
        Additional resources used in the conversion process.  Allows
        preprocessors to pass variables into the Jinja engine.
    index : int
        Index of the cell being processed
    """

    @functools.wraps(function)
    def wrappedfunc(nb, resources):
        get_logger().debug("Applying preprocessor: %s", function.__name__)
        for index, cell in enumerate(nb.cells):
            nb.cells[index], resources = function(cell, resources, index)
        return nb, resources

    return wrappedfunc


cr_pat = re.compile(r".*\r(?=[^\n])")


@cell_preprocessor
def coalesce_streams(cell, resources, index):
    """
    Merge consecutive sequences of stream output into single stream
    to prevent extra newlines inserted at flush calls

    Parameters
    ----------
    cell : NotebookNode cell
        Notebook cell being processed
    resources : dictionary
        Additional resources used in the conversion process.  Allows
        transformers to pass variables into the Jinja engine.
    index : int
        Index of the cell being processed
    """

    outputs = cell.get("outputs", [])
    if not outputs:
        return cell, resources

    last = outputs[0]
    new_outputs = [last]
    for output in outputs[1:]:
        if (
            output.output_type == "stream"
            and last.output_type == "stream"
            and last.name == output.name
        ):
            last.text += output.text

        else:
            new_outputs.append(output)
            last = output

    # process \r characters
    for output in new_outputs:
        if output.output_type == "stream" and "\r" in output.text:
            output.text = cr_pat.sub("", output.text)

    cell.outputs = new_outputs
    return cell, resources
