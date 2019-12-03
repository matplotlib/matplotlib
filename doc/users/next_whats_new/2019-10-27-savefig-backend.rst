``savefig()`` gained a ``backend`` keyword argument
---------------------------------------------------

The ``backend`` keyword argument to ``savefig`` can now be used to pick the
rendering backend without having to globally set the backend; e.g. one can save
pdfs using the pgf backend with ``savefig("file.pdf", backend="pgf")``.
