# Makefile for matplotlib

PYTHON = `which python`
VERSION = `${PYTHON} setup.py --version`

DISTFILES = API_CHANGES KNOWN_BUGS INSTALL README TODO license	\
	CHANGELOG Makefile INTERACTIVE			\
	MANIFEST.in lib lib/matplotlib lib/dateutil lib/pytz examples setup.py

RELEASE = matplotlib-${VERSION}


clean: 
	${PYTHON} setup.py clean;\
	rm -f *.png *.ps *.eps *.svg *.jpg *.pdf
	find . -name "_tmp*.py" | xargs rm -f;\
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;\
	find examples \( -name "*.svg" C-o -name "*.png" -o -name "*.pdf" -o -name "*.ps"  -o -name "*.eps" -o -name "*.tar" -o  -name "*.gz" -o -name "*.log" -o -name "*.aux" -o -name "*.tex" \) | xargs rm -f
	find unit \( -name "*.png" -o -name "*.ps"  -o -name "*.pdf" -o -name "*.eps" \) | xargs rm -f
	find . \( -name "#*" -o -name ".#*" -o -name ".*~" -o -name "*~" \) | xargs rm -f


release: ${DISTFILES}
	rm -f MANIFEST;\
	${PYTHON} license.py ${VERSION} license/LICENSE;\
	${PYTHON} setup.py sdist --formats=gztar,zip;

pyback: 
	tar cvfz pyback.tar.gz *.py lib src examples/*.py  unit/*.py 






