Changes for 0.63
================

.. code-block:: text

  Dates are now represented internally as float days since 0001-01-01,
  UTC.

  All date tickers and formatters are now in matplotlib.dates, rather
  than matplotlib.tickers

  converters have been abolished from all functions and classes.
  num2date and date2num are now the converter functions for all date
  plots

  Most of the date tick locators have a different meaning in their
  constructors.  In the prior implementation, the first argument was a
  base and multiples of the base were ticked.  e.g.,

    HourLocator(5)  # old: tick every 5 minutes

  In the new implementation, the explicit points you want to tick are
  provided as a number or sequence

     HourLocator(range(0,5,61))  # new: tick every 5 minutes

  This gives much greater flexibility.  I have tried to make the
  default constructors (no args) behave similarly, where possible.

  Note that YearLocator still works under the base/multiple scheme.
  The difference between the YearLocator and the other locators is
  that years are not recurrent.


  Financial functions:

    matplotlib.finance.quotes_historical_yahoo(ticker, date1, date2)

     date1, date2 are now datetime instances.  Return value is a list
     of quotes where the quote time is a float - days since gregorian
     start, as returned by date2num

     See examples/finance_demo.py for example usage of new API
