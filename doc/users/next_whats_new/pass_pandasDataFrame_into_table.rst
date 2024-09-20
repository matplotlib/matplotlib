New Feature - passing Pandas.DataFrame into ax.table(...)
----------------------------------------------------------

Pandas.DataFrame objects can now be used to add a tables to an axes. The cellText argument of
matplotlib.table.table was modified to accept it.

.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd

    data = {
        'Letter': ['A', 'B', 'C'],
        'Number': [100, 200, 300]
    }

    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    table = ax.table(df, loc='center')
    ax.axis('off')
    plt.show()
