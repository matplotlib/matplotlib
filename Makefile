# Makefile for matplotlib
# Copyright (C) 2003  <jdhunter@ace.bsd.uchicago.edu>
# $Header$
# $Log$
# Revision 1.6  2003/09/30 16:15:33  jdh2358
# added legend
#
# Revision 1.5  2003/09/22 14:04:46  jdh2358
# small bugfixes
#
# Revision 1.4  2003/09/15 17:54:16  jdh2358
# multiple backed support take II
#
# Revision 1.3  2003/05/12 19:56:54  jdh2358
# update license to version 2 and the docs
#
# Revision 1.2  2003/05/12 15:53:48  jdh2358
# update matplotlib
#
# Revision 1.1  2003/05/12 15:50:11  jdh2358
# adding Makefile, releases, docs
#

PYTHON = /usr/bin/python2.2
PYDOC = /usr/bin/pydoc
VERSION = `${PYTHON} setup.py --version`
DISTFILES = INSTALL README TODO LICENSE CHANGELOG Makefile GOALS INTERACTIVE \
	MANIFEST.in matplotlib examples setup.py
MODULES = matplotlib.afm matplotlib.axes matplotlib.artist		\
	matplotlib.backend_bases matplotlib.cbook matplotlib.lines	\
	matplotlib.patches matplotlib.matlab matplotlib.mlab		\
	matplotlib.backends.backend_gtk matplotlib.backends.backend_gd	\
	matplotlib.backends.backend_ps 	matplotlib.backends.backend_template
RELEASE = matplotlib-${VERSION}


clean: 
	${PYTHON} setup.py clean;\
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;\
	find examples -name "*.png"  | xargs rm -f;\
	find matplotlib -name "*.png"  | xargs rm -f;

htmldocs: 
	rm -f docs/*.html;\
	${PYDOC} -w ${MODULES};\
	mv *.html docs/

release: ${DISTFILES}
	rm -rf ${RELEASE};\
	mkdir ${RELEASE};\
	cp -a ${DISTFILES}  ${RELEASE}/;\
	rm -rf ${RELEASE}/CVS ${RELEASE}/matplotlib/CVS ${RELEASE}/examples/CVS ${RELEASE}/examples/figures/* ${RELEASE}/examples/*.png ${RELEASE}/examples/*.pyc ${RELEASE}/matplotlib/*.png ${RELEASE}/matplotlib/*.pyc ;\
	tar cvfz releases/${RELEASE}.tar.gz  ${RELEASE}/ --dereference;\
	python setup.py bdist_wininst;\
	cp dist/${RELEASE}.win32.exe releases/;\
	zip -r ${RELEASE}.zip ${RELEASE};\
	mv ${RELEASE}.zip releases/;\
	rm -rf ${RELEASE};

