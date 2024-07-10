``svg.id`` rcParam
~~~~~~~~~~~~~~~~~~
The value of this new rcParam controls whether the top-level ``<svg>`` tag
contains an ``id`` attribute and what its value is. When set to ``None`` (the
default), no ``id`` tag is included

.. code-block:: XML

    <svg
        xmlns:xlink="http://www.w3.org/1999/xlink"
        width="50pt" height="50pt"
        viewBox="0 0 50 50"
        xmlns="http://www.w3.org/2000/svg"
        version="1.1"
        id="svg1"
    ></svg>

This is useful if you would like to link the entire matplotlib SVG file within
another SVG file with the ``<use>`` tag.

.. code-block:: XML

    <svg>
    <use
        width="50" height="50"
        xlink:href="mpl.svg#svg1" id="use1"
        x="0" y="0"
    /></svg>

Where the ``#svg1`` indicator will now refer to the top level ``<svg>`` tag, and
will hence result in the inclusion of the entire file.
