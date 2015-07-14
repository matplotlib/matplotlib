# Makefile for matplotlib

PYTHON = `which python`
VERSION = `${PYTHON} setup.py --version`

DISTFILES = API_CHANGES KNOWN_BUGS INSTALL README license	\
	CHANGELOG Makefile INTERACTIVE			\
	MANIFEST.in lib lib/matplotlib lib/dateutil lib/pytz examples setup.py

RELEASE = matplotlib-${VERSION}


clean:
	${PYTHON} setup.py clean;\
	rm -f *.png *.ps *.eps *.svg *.jpg *.pdf
	find . -name "_tmp*.py" | xargs rm -f;\
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;\
	find unit \( -name "*.png" -o -name "*.ps"  -o -name "*.pdf" -o -name "*.eps" \) | xargs rm -f
	find . \( -name "#*" -o -name ".#*" -o -name ".*~" -o -name "*~" \) | xargs rm -f


release: ${DISTFILES}
	rm -f MANIFEST;\
	${PYTHON} license.py ${VERSION} license/LICENSE;\
	${PYTHON} setup.py sdist --formats=gztar,zip;

pyback:
	tar cvfz pyback.tar.gz *.py lib src examples/*.py  unit/*.py


_build_osx105:
	CFLAGS="-Os -arch i386 -arch ppc" LDFLAGS="-Os -arch i386 -arch ppc" python setup.py build

build_osx105:
	echo "Use 'make -f fetch deps mpl_install instead'"


jdh_doc_snapshot:
	git pull;\
	python setup.py install --prefix=~/dev;\
	cd doc;\
	rm -rf build;\
	python make.py clean;\
	python make.py html latex sf sfpdf;


test:
	${PYTHON} setup.py test


test-coverage:
	${PYTHON} setup.py test --with-coverage --cover-package=matplotlib


