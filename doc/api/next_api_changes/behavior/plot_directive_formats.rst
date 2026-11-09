The plot directive only renders the formats a build can use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``.. plot::`` directive, i.e., ``matplotlib.sphinxext.plot_directive``
saved every entry of ``plot_formats`` on every build. With
``plot_html_show_formats = False``, that meant an HTML build also wrote a PDF
of each figure, which was not linked to. The directive now checks the Sphinx
builder before rendering. For HTML, it renders the first format, and the rest
of the formats are rendered only if ``plot_html_show_formats`` is True.
Other Sphinx builders get formats in their `supported_image_types`_, i.e.,
LaTeX builder retains the rendering of its PDFs.

Projects that picked ``plot_formats`` from ``sys.argv`` to get the same
effect can drop doing so. Files for skipped formats no longer appear in
the ``plot_directive`` build directory.

.. _supported_image_types: https://www.sphinx-doc.org/en/master/extdev/builderapi.html#sphinx.builders.Builder.supported_image_types
