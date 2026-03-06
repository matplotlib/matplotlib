``figure.titlelocation`` rcParam
--------------------------------

A new :rc:`figure.titlelocation` rcParam has been added to control the
default horizontal alignment of the figure suptitle, analogous to the
existing :rc:`axes.titlelocation` for axes titles.  Supported values are
``'left'``, ``'center'`` (default), and ``'right'``.

.. code-block:: python

    import matplotlib as mpl
    mpl.rcParams['figure.titlelocation'] = 'left'

    fig, ax = plt.subplots()
    fig.suptitle('Left-aligned suptitle')
