API Changes for 3.10.9
======================


Deprecations
------------


Arbitrary code in ``axes.prop_cycle`` rcParam strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``axes.prop_cycle`` rcParam accepts Python expressions that are evaluated
in a limited context.  The evaluation context has been further limited and some
expressions that previously worked (list comprehensions, for example) no longer
will. This change is made without a deprecation period to improve security.
The previously documented cycler operations at
https://matplotlib.org/cycler/ are still supported.

This change was originally slated for v3.11.0 of Matplotlib, but was additionally
backported due to the security implications.
