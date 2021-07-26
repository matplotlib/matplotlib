Type 42 Subsetting is now enabled for PDF/PS backends
-----------------------------------------------------

`~matplotlib.backends.backend_pdf` and `~matplotlib.backends.backend_ps` now use
a unified Type 42 font subsetting interface, with the help of `fontTools <https://fonttools.readthedocs.io/en/latest/>`_

Set `~matplotlib.RcParams`'s *fonttype* value as ``42`` to trigger this workflow:

.. code-block::

    # for PDF backend
    plt.rcParams['pdf.fonttype'] = 42

    # for PS backend
    plt.rcParams['ps.fonttype'] = 42


    fig, ax = plt.subplots()
    ax.text(0.4, 0.5, 'subsetted document is smaller in size!')

    fig.savefig("document.pdf")
    fig.savefig("document.ps")
