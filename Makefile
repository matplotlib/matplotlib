# Makefile for matplotlib
# Copyright (C) 2003  <jdhunter@ace.bsd.uchicago.edu>
# $Header$
# $Log$
# Revision 1.12  2004/01/26 18:27:46  jdh2358
# more ps and text API fixes
#
# Revision 1.11  2003/11/19 16:45:09  jdh2358
# updated plotting commands list
#
# Revision 1.10  2003/11/14 00:07:29  jdh2358
# added log transforms to new API
#
# Revision 1.9  2003/11/06 23:09:53  jdh2358
# fixed some problems with the file release system
#
# Revision 1.8  2003/10/23 15:42:43  jdh2358
# fixed figure text clip bug
#
# Revision 1.7  2003/10/18 17:54:26  jdh2358
# fixed interactive2 and several small bugs
#
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

PYTHON = /usr/local/bin/python2.3
PYDOC = /usr/local/bin/pydoc
VERSION = `${PYTHON} setup.py --version`

DISTFILES = API_CHANGES KNOWN_BUGS INSTALL README TODO LICENSE	\
	CHANGELOG Makefile GOALS INTERACTIVE			\
	MANIFEST.in matplotlib examples setup.py

MODULES =                                       \
        matplotlib.afm				\
	matplotlib.axis				\
	matplotlib.axes				\
	matplotlib.artist			\
	matplotlib.backend_bases		\
	matplotlib.cbook			\
	matplotlib.legend			\
	matplotlib.lines			\
	matplotlib.patches			\
	matplotlib.matlab			\
	matplotlib.mlab				\
	matplotlib.text				\
	matplotlib.transforms			\
	matplotlib.backends.backend_gtk		\
	matplotlib.backends.backend_gd		\
	matplotlib.backends.backend_ps		\
	matplotlib.backends.backend_template	\
        matplotlib.backends.backend_wx

RELEASE = matplotlib-${VERSION}


clean: 
	${PYTHON} setup.py clean;\
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;\
	find examples -name "*.png"  | xargs rm -f;\
	find matplotlib -name "*.png"  | xargs rm -f;

htmldocs: 
	rm -rf htdocs/matplotlib;\
	cp -a matplotlib htdocs/;\
	cp examples/*.py htdocs/examples;\
	rm -f docs/*.html;\
	${PYDOC} -w ${MODULES};\
	mv *.html docs/
	cd htdocs;\
	${PYTHON} process_docs.py;\
	${PYTHON} convert.py;\
	tar cfz site.tar.gz *.html screenshots tut examples

release: ${DISTFILES}
	${PYTHON} license.py ${VERSION};\
	${PYTHON} setup.py sdist --formats=gztar,zip;\
	${PYTHON} setup.py bdist_wininst;


pyback: 
	tar cvfz pyback.tar.gz *.py matplotlib/*.py examples/*.py matplotlib/backends/*.py unit/*.py