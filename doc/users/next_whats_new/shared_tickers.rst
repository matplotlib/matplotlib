Two shared axis can have different tick formatters and locators
---------------------------------------------------------------

Previously two shared axis were forced to have the same tick formatter and
tick locator. It is now possible to set shared axis to have different tickers
and formatters using the *share_tickers* keyword argument to `twinx()` and
`twiny()`.
