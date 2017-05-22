Semantic Data for SVG
---------------------

The SVG backend now supports embedding additional semantic data in the generated
XML. :func:`~matplotlib.pyplot.savefig` and :meth:`~matplotlib.figure.Figure.savefig`
Now accepts ``svg_gid_data`` as a keyword argument expecting a :class:`dict`-like
object with string keys and :class:`dict` values. For each Artist, if the result of
:meth:`~matplotlib.artist.Artist.get_gid` is found in ``svg_gid_data``, the value :class:`dict`
will be included as attributes of the Artist's container element.

Additionally these functions also now accept ``svg_attribs`` keyword argument also
taking a :class:`dict` object to add and override attributes on the top-level ``<svg>``
element. A special key in this dictionary, ``"extra_content"`` will be included as arbitrary
XML directly under the top-level ``<svg>``.


Example
```````

.. code-block:: python

    plt.savefig("test.svg", format='svg', svg_attribs={"class": "svg-figure"},
                svg_gid_data={"target-geom": {
                    "class": "target",
                    "data-charge-level": 77.5,
                    "onclick": "alert(this.dataset.chargeLevel)"
                }})

produces

.. code-block:: xml

    <svg class="svg-figure">
        <g id="target-geom" data-charge-level="77.5" onclick="alert(this.dataset.chargeLevel)">
            ...
        </g>
    </svg>
