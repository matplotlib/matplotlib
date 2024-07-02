********************
``matplotlib.dates``
********************

Matplotlib provides sophisticated date plotting capabilities, standing on the
shoulders of python :mod:`datetime` and the add-on module dateutil_.

.. currentmodule:: matplotlib.dates

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: entry
   :class: multicol-toc

.. automodule:: matplotlib.dates
   :no-members:
   :no-undoc-members:

Overview
========

By default, Matplotlib uses the units machinery described in
`~matplotlib.units` to convert `datetime.datetime`, `datetime.timedelta`,
`numpy.datetime64` or `numpy.timedelta64` objects when plotted on an
x- or y-axis. The user does not need to do anything for dates to be formatted,
but dates often have strict formatting needs, so this module provides many
tick locators and formatters. A basic example using `numpy.datetime64` is

.. plot::
   :include-source:

   import numpy as np
   times = np.arange(np.datetime64('2001-01-02'),
                     np.datetime64('2002-02-03'), np.timedelta64(1, 'D'))
   y = np.random.randn(len(times))
   fig, ax = plt.subplots()
   ax.plot(times, y)
   fig.autofmt_xdate()


If you want to customize the tick labels that are shown on the x-axis, you
need to manually configure the tick locator and formatter for this axis.
The tick locator determines how many ticks are shown and in which interval they
are shown.
The formatter defines how a date or timedelta value is represented as a string.
This means that tick formatting is independent of the number, interval and
location of ticks.

By default, Matplotlib uses automatic tick locators and formatters for
date and timedelta values. This means that a reasonable number of ticks is
shown and the tick label is formatted in such a way that includes only the
relevant information to keep it as short as possible.


.. seealso::

    - :doc:`/gallery/text_labels_and_annotations/date`
    - :doc:`/gallery/ticks/date_concise_formatter`
    - :doc:`/gallery/ticks/date_demo_convert`


.. _date-format:

Matplotlib date format
======================

Matplotlib represents dates using floating point numbers specifying the number
of days since a default epoch of 1970-01-01 UTC; for example,
1970-01-01, 06:00 is the floating point number 0.25. The formatters and
locators require the use of `datetime.datetime` objects, so only dates between
year 0001 and 9999 can be represented.  Microsecond precision
is achievable for (approximately) 70 years on either side of the epoch, and
20 microseconds for the rest of the allowable range of dates (year 0001 to
9999). The epoch can be changed at import time via `set_epoch` or
:rc:`dates.epoch` to other dates if necessary; see
:doc:`/gallery/ticks/date_precision_and_epochs` for a discussion.

.. note::

   Before Matplotlib 3.3, the epoch was 0000-12-31 which lost modern
   microsecond precision and also made the default axis limit of 0 an invalid
   datetime.  In 3.3 the epoch was changed as above.  To convert old
   ordinal floats to the new epoch, users can do::

     new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))


There are a number of helper functions to convert between :mod:`datetime`
objects and Matplotlib dates:

.. currentmodule:: matplotlib.dates

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   datestr2num
   date2num
   num2date
   drange
   set_epoch
   get_epoch


.. note::

   Like Python's `datetime.datetime`, Matplotlib uses the Gregorian calendar
   for all conversions between dates and floating point numbers. This practice
   is not universal, and calendar differences can cause confusing
   differences between what Python and Matplotlib give as the number of days
   since 0001-01-01 and what other software and databases yield.  For
   example, the US Naval Observatory uses a calendar that switches
   from Julian to Gregorian in October, 1582.  Hence, using their
   calculator, the number of days between 0001-01-01 and 2006-04-01 is
   732403, whereas using the Gregorian calendar via the datetime
   module we find::

     In [1]: date(2006, 4, 1).toordinal() - date(1, 1, 1).toordinal()
     Out[1]: 732401

All the Matplotlib date converters, locators and formatters are timezone aware.
If no explicit timezone is provided, :rc:`timezone` is assumed, provided as a
string.  If you want to use a different timezone, pass the *tz* keyword
argument of `num2date` to any date tick locators or formatters you create. This
can be either a `datetime.tzinfo` instance or a string with the timezone name
that can be parsed by `~dateutil.tz.gettz`.

A wide range of specific and general purpose date tick locators and
formatters are provided in this module.  See
:mod:`matplotlib.ticker` for general information on tick locators
and formatters.  These are described below.

The dateutil_ module provides additional code to handle date ticking, making it
easy to place ticks on any kinds of dates.  See examples below.

.. _dateutil: https://dateutil.readthedocs.io


Matplotlib timedelta format
===========================

Matplotlib represents timedeltas using floating point numbers specifying the
number of days, similar to how dates are represented. For example, a timedelta
of 1 day, 06:00 is the floating point number 1.25. The formatters and tick
locators require the use of `datetime.timedelta` objects, therefore, only
timedeltas up to +-999999999 days are supported.
Microsecond precision is achievable for (approximately) +-70 years.

There are two of helper functions to convert between `~datetime.timedelta`
objects and Matplotlib timedeltas. Additionally, Matplotlib defines a
`strftimedelta` function. This is the timedelta equivalent to
`datetime.date.strftime`, but the `datetime` module does not define such a
function for timedeltas.
The format codes for `strftimedelta` are similar to those used for
`~datetime.date.strftime`. The complete reference is given in the documentation
for `strftimedelta`.

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   num2timedelta
   timedelta2num
   strftimedelta
   strftdnum


Tick Locators
=============

Tick locators determine how many ticks are shown on an `~matplotlib.axis.Axis`
and where they are located. Most tick locators create ticks in regular
intervals.

.. inheritance-diagram::
   rrulewrapper
   DateLocator
   AutoDateLocator
   RRuleLocator
   YearLocator
   WeekdayLocator
   DayLocator
   HourLocator
   MinuteLocator
   SecondLocator
   MicrosecondLocator
   TimedeltaLocator
   AutoTimedeltaLocator
   :parts: 1
   :top-classes: matplotlib.ticker.Locator


.. _date-locators:

Date tick locators
------------------

Most of the date tick locators can locate single or multiple ticks. For example::

    # import constants for the days of the week
    from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

    # tick on Mondays every week
    loc = WeekdayLocator(byweekday=MO, tz=tz)

    # tick on Mondays and Saturdays
    loc = WeekdayLocator(byweekday=(MO, SA))

In addition, most of the constructors take an interval argument::

    # tick on Mondays every second week
    loc = WeekdayLocator(byweekday=MO, interval=2)

The rrule locator allows completely general date ticking::

    # tick every 5th easter
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)


.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   AutoDateLocator
   YearLocator
   MonthLocator
   WeekdayLocator
   DayLocator
   HourLocator
   MinuteLocator
   SecondLocator
   MicrosecondLocator
   DateLocator
   RRuleLocator
   rrulewrapper


Timedelta tick locators
-----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   TimedeltaLocator
   AutoTimedeltaLocator


Formatters
==========

Formatters define the format of the tick label.

.. _date-formatters:

Date formatters
---------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   AutoDateFormatter
   ConciseDateFormatter
   DateFormatter

The automatic date formatters `AutoDateFormatter` and `ConciseDateFormatter`
are most useful when used with the `AutoDateLocator`.


Timedelta formatters
--------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   AutoTimedeltaFormatter
   ConciseTimedeltaFormatter
   TimedeltaFormatter

The automatic date formatters `AutoTimedeltaFormatter` and
`ConciseTimedeltaFormatter` are most useful when used with the
`AutoTimedeltaLocator`.


Conversion Interface
====================

.. inheritance-diagram::
   DateConverter
   ConciseDateConverter
   TimedeltaConverter
   ConciseTimedeltaConverter
   :parts: 1
   :top-classes: matplotlib.ticker.Formatter


.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   DateConverter
   ConciseDateConverter
   TimedeltaConverter
   ConciseTimedeltaConverter
