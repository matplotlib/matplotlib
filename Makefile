# Makefile for matplotlib
# Copyright (C) 2003  <jdhunter@ace.bsd.uchicago.edu>
# $Header$
# $Log$
# Revision 1.29  2004/04/21 15:28:55  jdh2358
# sync to 0.53
#
# Revision 1.28  2004/04/20 22:53:37  jdh2358
# removed tz info from dates; updated htdocs
#
# Revision 1.27  2004/04/20 11:32:35  jdh2358
# added finance module
#
# Revision 1.26  2004/03/16 20:26:36  jdh2358
# sync to 0.52 release
#
# Revision 1.25  2004/03/15 14:49:46  jdh2358
# updated image to work with data coords; moved image extendsion to _image
#
# Revision 1.24  2004/03/08 14:56:44  jdh2358
# fixed subplot bug
#
# Revision 1.23  2004/02/27 14:47:46  jdh2358
# added matplotlibrc and tk win32 compile fixes
#
# Revision 1.22  2004/02/18 18:11:24  jdh2358
# updated setepext to work with darwin
#
# Revision 1.21  2004/02/17 15:10:43  jdh2358
# updating to 0.50
#
# Revision 1.20  2004/02/16 18:04:55  jdh2358
# fexed wx to work on windows and linux
#
# Revision 1.19  2004/02/11 19:16:01  jdh2358
# reorganized table
#
# Revision 1.18  2004/02/11 00:07:44  jdh2358
# relocated gtkgd ext mod
#
# Revision 1.17  2004/01/30 21:33:19  jdh2358
# last changes for 0.50e
#
# Revision 1.16  2004/01/30 19:58:53  jdh2358
# update...
#
# Revision 1.15  2004/01/30 18:20:55  jdh2358
# renamed backend_gtk2 to backend_gtkgd
#
# Revision 1.14  2004/01/29 19:26:46  jdh2358
# added API_CHANGES to htdocs
#
# Revision 1.13  2004/01/27 16:18:08  jdh2358
# updated README and INSTALL
#
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
VERSION = `${PYTHON} setup.py --version`

DISTFILES = API_CHANGES KNOWN_BUGS INSTALL README TODO license	\
	CHANGELOG Makefile GOALS INTERACTIVE			\
	MANIFEST.in matplotlib examples setup.py

RELEASE = matplotlib-${VERSION}


clean: 
	${PYTHON} setup.py clean;\
	find . -name "_tmp*.py" | xargs rm -f;\
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;\
	find examples \( -name "*.png" -o -name "*.ps"  -o -name "*.jpg" -o -name "*.eps" \) | xargs rm -f
	find . \( -name "#*" -o -name ".#*" -o -name ".*~" -o -name "*~" \) | xargs rm -f


release: ${DISTFILES}
	${PYTHON} license.py ${VERSION};\
	${PYTHON} setup.py sdist --formats=gztar,zip;

pyback: 
	tar cvfz pyback.tar.gz *.py matplotlib/*.py examples/*.py matplotlib/backends/*.py unit/*.py

htmldocs: 

	cp examples/*.py htdocs/examples;\
	cp .matplotlibrc NUMARRAY_ISSUES API_CHANGES htdocs/;\
	cd htdocs;\
	${PYTHON} process_docs.py;\
	${PYTHON} convert.py;\
	tar cfz site.tar.gz *.html screenshots tut examples gd .matplotlibrc NUMARRAY_ISSUES  API_CHANGES;\
	cd ..;\
	cp -a matplotlib htdocs/;