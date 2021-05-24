
Simplifying the font setting for usetex mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now the :rc:`font.family` accepts some font names as value for a more
user-friendly setup.

.. code-block::

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })