Metadata savefig kwarg
----------------------

:func:`~matplotlib.pyplot.savefig` now accepts `metadata` as a keyword argument.
It can be used to store key/value pairs in the image metadata.

Supported formats and backends
``````````````````````````````
* 'png' with Agg backend
* 'pdf' with PDF backend (see
  :func:`~matplotlib.backends.backend_pdf.PdfFile.writeInfoDict` for a list of
  supported keywords)
* 'eps' and 'ps' with PS backend (only 'Creator' key is accepted)

Example
```````
::

    plt.savefig('test.png', metadata={'Software': 'My awesome software'})

