Cleanup of stock sample data
````````````````````````````

The sample data of stocks has been cleaned up to remove redundancies and
increase portability. The ``AAPL.dat.gz``, ``INTC.dat.gz`` and ``aapl.csv``
files have been removed entirely and will also no longer be available from
`matplotlib.cbook.get_sample_data`. If a CSV file is required, we suggest using
the ``msft.csv`` that continues to be shipped in the sample data. If a NumPy
binary file is acceptable, we suggest using one of the following two new files.
The ``aapl.npy.gz`` and ``goog.npy`` files have been replaced by ``aapl.npz``
and ``goog.npz``, wherein the first column's type has changed from
`datetime.date` to `np.datetime64` for better portability across Python
versions. Note that matplotlib does not fully support `np.datetime64` as yet.
