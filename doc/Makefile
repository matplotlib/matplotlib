# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = matplotlib
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# workaround because sphinx does not completely clean up (#11139)
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf "$(SOURCEDIR)/build"
	rm -rf "$(SOURCEDIR)/api/_as_gen"
	rm -rf "$(SOURCEDIR)/gallery"
	rm -rf "$(SOURCEDIR)/plot_types"
	rm -rf "$(SOURCEDIR)/tutorials"
	rm -rf "$(SOURCEDIR)/users/explain"
	rm -rf "$(SOURCEDIR)/savefig"
	rm -rf "$(SOURCEDIR)/sphinxext/__pycache__"
	rm -f $(SOURCEDIR)/_static/constrained_layout*.png
	rm -f $(SOURCEDIR)/sg_execution_times.rst

show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(shell pwd)/build/html/index.html')"

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# This will skip the subdirectories listed in .mpl_skip_subdirs.yaml If
# this file does not exist, one will be created for you.  This option useful
# to quickly build parts of the docs, but the resulting build will not
# have all the crosslinks etc.
html-skip-subdirs:
	$(SPHINXBUILD) -D skip_sub_dirs=1 -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
