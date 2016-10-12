.. _matplotlib-for-dev:

=================================================
Installing matplotlib from source for development
=================================================

After obtaining a local copy of the matpotlib source code (:ref:`set-up-fork`),
navigate to the matplotlib directory and run the following in the shell:

::
    
    python setup.py develop

or::
  
   pip install -v -e .

This installs matplotlib for development (i.e., builds everything and places
the symbolic links back to the source code). This command has to be called
everytime a `c` file is changed. Note that changing branches may change the
`c`-code.

When working on bleeding edge packages, setting up a
`virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_ or a `conda
environment <http://conda.pydata.org/docs/using/envs.html>`_ is recommended.

