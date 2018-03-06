Multipage PDF support for pgf backend
-------------------------------------

The pgf backend now also supports multipage PDF files.

.. code-block:: python

    from matplotlib.backends.backend_pgf import PdfPages
    import matplotlib.pyplot as plt

    with PdfPages('multipage.pdf') as pdf:
        # page 1
        plt.plot([2, 1, 3])
        pdf.savefig()

        # page 2
        plt.cla()
        plt.plot([3, 1, 2])
        pdf.savefig()
