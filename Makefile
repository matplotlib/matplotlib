# Makefile for matplotlib
# Copyright (C) 2003  <jdhunter@ace.bsd.uchicago.edu>
# $Header$
# $Log$
# Revision 1.3  2003/05/12 19:56:54  jdh2358
# update license to version 2 and the docs
#
# Revision 1.2  2003/05/12 15:53:48  jdh2358
# update matplotlib
#
# Revision 1.1  2003/05/12 15:50:11  jdh2358
# adding Makefile, releases, docs
#

VERSION = `python setup.py --version`
DISTFILES = INSTALL README TODO LICENSE CHANGELOG Makefile GOALS INTERACTIVE \
	    matplotlib examples examples setup.py
MODULES = artist cbook gtkutils lines patches colors text matlab figure
RELEASE = matplotlib-${VERSION}


clean: 
	python setup.py clean;
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;
	find examples -name "*.png"  | xargs rm -f;
	find matplotlib -name "*.png"  | xargs rm -f;

htmldocs: 
	rm -f docs/*.html;\
	cd matplotlib;\
	pydoc -w ${MODULES};\
	mv *.html ../docs/

release: ${DISTFILES}
	rm -rf ${RELEASE};
	mkdir ${RELEASE};
	cp -a ${DISTFILES}  ${RELEASE}/;
	tar cvfz releases/${RELEASE}.tar.gz  ${RELEASE}/ --dereference;
	python setup.py bdist_wininst;
	cp dist/${RELEASE}.win32.exe releases/;
	rm -rf ${RELEASE};

