``ax.table`` will accept a pandas dataframe
--------------------------------------------

The `~Axes.axes.table` method can now accept a data frame for the ``cellText`` method, which
it attempts to render with column headers set by ``df.columns.to_numpy()`` and cell data set by ``df.to_numpy()``.

.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd

    data = {
        'Letter': ['A', 'B', 'C'],
        'Number': [100, 200, 300]
    }

    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    table = ax.table(df, loc='center')  # or table = ax.table(cellText=df, loc='center')
    ax.axis('off')
    plt.show()
