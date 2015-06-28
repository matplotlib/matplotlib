cbook.is_sequence_of_strings recognizes string objects
``````````````````````````````````````````````````````

This is primarily how pandas stores a sequence of strings ::

    import pandas as pd
    import matplotlib.cbook as cbook

    a = np.array(['a', 'b', 'c'])
    print(cbook.is_sequence_of_strings(a))  # True

    a = np.array(['a', 'b', 'c'], dtype=object)
    print(cbook.is_sequence_of_strings(a))  # True

    s = pd.Series(['a', 'b', 'c'])
    print(cbook.is_sequence_of_strings(s))  # True

Previously, the last two prints returned false.
