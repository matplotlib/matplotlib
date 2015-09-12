Working with labeled data like pandas DataFrames
------------------------------------------------
Plot methods which take arrays as inputs can now also work with labeled data
and unpack such data.

This means that the following two examples produce the same plot:

Example ::

    df = pandas.DataFrame({"var1":[1,2,3,4,5,6], "var2":[1,2,3,4,5,6]})
    plt.plot(df["var1"], df["var2"])


Example ::

    plt.plot("var1", "var2", data=df)

This works for most plotting methods, which expect arrays/sequences as
inputs.  ``data`` can be anything which supports ``__getitem__``
(``dict``, ``pandas.DataFrame``, ``h5py``, ...) to access ``array`` like
values with string keys.

In addition to this, some other changes were made, which makes working with
labeled data (ex ``pandas.Series``) easier:

* For plotting methods with ``label`` keyword argument, one of the
  data inputs is designated as the label source.  If the user does not
  supply a ``label`` that value object will be introspected for a
  label, currently by looking for a ``name`` attribute.  If the value
  object does not have a ``name`` attribute but was specified by as a
  key into the ``data`` kwarg, then the key is used.  In the above
  examples, this results in an implicit ``label="var2"`` for both
  cases.

* ``plot()`` now uses the index of a ``Series`` instead of
  ``np.arange(len(y))``, if no ``x`` argument is supplied.
