API Changes for 3.3.1
=====================

Deprecations
------------

Reverted deprecation of ``num2epoch`` and ``epoch2num``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These two functions were deprecated in 3.3.0, and did not return
an accurate Matplotlib datenum relative to the new Matplotlib epoch
handling (`~.dates.get_epoch` and :rc:`date.epoch`).  This version
reverts the deprecation.

Functions ``epoch2num`` and ``dates.julian2num`` use ``date.epoch`` rcParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now ``epoch2num`` and (undocumented) ``julian2num`` return floating point
days since `~.dates.get_epoch` as set by :rc:`date.epoch`, instead of
floating point days since the old epoch of "0000-12-31T00:00:00".  If
needed, you can translate from the new to old values as
``old = new + mdates.date2num(np.datetime64('0000-12-31'))``
