# Makefile for matplotlib
# Copyright (C) 2003  <jdhunter@ace.bsd.uchicago.edu>
# $Header$
# $Log$
# Revision 1.1  2003/05/12 15:50:11  jdh2358
# adding Makefile, releases, docs
#

VERSION = `python setup.py --version`
DISTFILES = INSTALL README TODO LICENSE CHANGELOG \
	    matplotlib examples examples setup.py
MODULES = artist cbook gtkutils lines patches colors text matlab figure
RELEASE = matplotlib-${VERSION}


clean: 
	python setup.py clean;
	rm -rf build dist;
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;
	find examples -name "*.png"  | xargs rm -f;
	find matplotlib -name "*.png"  | xargs rm -f;

htmldocs: 
	rm -f docs/*;
	cd matplotlib;
	pydoc -w ${MODULES};
	mv *.html ../docs/

release: ${DISTFILES}
	rm -rf ${RELEASE};
	mkdir ${RELEASE};
	cp -a ${DISTFILES}  ${RELEASE}/;
	tar cvfz releases/${RELEASE}.tar.gz  ${RELEASE}/ --dereference;
	python setup.py bdist_wininst;
	cp dist/${RELEASE}.win32.exe releases/;
	rm -rf ${RELEASE};

