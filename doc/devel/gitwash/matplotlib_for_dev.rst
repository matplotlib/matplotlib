.. _matplotlib-for-dev:

===================================================
    Install for matplotlib source for development
===================================================

After obtaining a local copy of the matpotlib source code (:ref:`set-up-fork`),
navigate to the matplotlib directory and run the following in the shell:

::
    
    python setup.py develop

This installs matplotlib for development (i.e., builds everything and places the
symbolic links back to the source code).

You may want to consider setting up a `virtual environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.
