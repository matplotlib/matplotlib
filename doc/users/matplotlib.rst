Matplotlib's Mission
====================

Matplotlib aims to be the foundational plotting library for the Scientific
Python Ecosystem.



It has been over a decade since John Hunter wrote :ref:`History` and the world
has changed.  Python is no longer a fringe language in science but is now one
of the most widely used languages in science and data science.  Python, and
Matplotlib, are now regularly taught in schools and used in research both at
the inididivual grad student scale to the Mars helicopter [CITE].  Data
structures that are now considered standard, such as pandas.DataFrame or
xarray.DataArray, did not yet exist.  Over that time there has also developed a
number of domain-specific plotting libraries extending Matplotlib.  The
packaging ecosystem in Python has gotten much better than it was a decade ago,
expecting users to install Matplotlib and several extensions is more reasonable
than it was.   In light of the changes to the world it is worth re-addressing
the Matplotlib's role in the wider Scientific Python Ecosystem.


To paraphrase from JDH's original requirements, Matplotlib should:

* Produce publication quality output (including vector formats);
* Be embeddedable in graphical user interfaces for application development;
* Code be understandable and extendable by scientists and data scientists;
* Support users across all fields of science and industry;
* Easy things should be easy;
* Hard things should be possible.

Unfortunately, these goals have inherent tension.  "Easy", while subjective and
context dependent, it can be understood as how little work the user has to do
to get from data to a visualization.  How much work the user has to do depends
on how well aligned the assumptions the library makes about what the user wants
to do.  However, given the broad range of domains where Matplotlib is used,
from time series analysis to statistics or from image analysis to mapping, it
is very easy to make assumptions that work well for one application that are
catastrophically wrong for another!  To all of our users we need to think of
Matplotlib as not just the core library, but also including the domain specific
plotting libraries built on and around us.


To this end, Matplotlib's role in the ecosystem is to provide the tools to build
the tools that users will use.
