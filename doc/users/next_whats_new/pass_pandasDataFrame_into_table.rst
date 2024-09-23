New Feature - passing Pandas.DataFrame into ax.table(...)
----------------------------------------------------------

Pandas.DataFrame objects can now be passed directly to the ax.table
plotting method via the cellText argument. See the following code
block for an example implementation.

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
