``ax.table`` will accept a pandas DataFrame
--------------------------------------------

The `~.axes.Axes.table` method can now accept a Pandas DataFrame for the ``cellText`` argument.

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
