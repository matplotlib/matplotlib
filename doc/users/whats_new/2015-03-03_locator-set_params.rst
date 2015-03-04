API Consistency fix within Locators set_params() function
---------------------------------------------------------

set_params() function, which sets parameters within a Locator type instance,
is now available to all Locator types. The implementation also prevents unsafe
usage by strictly defining the parameters that a user can set.

To use, simply call set_params() on a Locator instance with desired arguments:
::

    loc = matplotlib.ticker.LogLocator()
    # Set given attributes for loc.
    loc.set_params(numticks=8, numdecs=8, subs=[2.0], base=8)
    # The below will error, as there is no such parameter for LogLocator
    # named foo
    # loc.set_params(foo='bar')
