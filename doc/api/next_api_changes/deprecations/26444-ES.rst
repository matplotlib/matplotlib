Passing non-int or sequence of non-int to ``Table.auto_set_column_width``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Column numbers are ints, and formerly passing any other type was effectively
ignored. This will become an error in the future.
