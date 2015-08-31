Working with labeled data like pandas DataFrames
------------------------------------------------
Plot methods which take arrays as inputs can now also work with labeled data
and unpack such data.

This means that the following two examples produce the same plot::

Example ::
    df = pandas.DataFrame({"var1":[1,2,3,4,5,6], "var2":[1,2,3,4,5,6]})
    plt.plot(df["var1"], df["var2"])


Example ::
    plt.plot("var1", "var2", data=df)

This works for most plotting methods, which expect arrays/sequences as
inputs and ``data`` can be anything which supports ``__get_item__``
(``dict``, ``pandas.DataFrame``,...).

In addition to this, some other changes were made, which makes working with
``pandas.DataFrames`` easier:

* For plotting methods which understand a ``label`` keyword argument but the
  user does not supply such an argument, this is now implicitly set by either
  looking up ``.name`` of the right input or by using the label supplied to
  lookup the input in ``data``. In the above examples, this results in an
  implicit ``label="var2"`` for both cases.

* ``plot()`` now uses the index of a ``Series`` instead of
  ``np.arange(len(y))``, if no ``x`` argument is supplied.
