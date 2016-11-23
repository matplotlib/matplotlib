# Makefile for matplotlib

PYTHON ?= python

all: test

flake8-diff:
	./ci/travis/flake8_diff.sh

test:
	${PYTHON} tests.py

test-coverage:
	python tests.py

clean:
	${PYTHON} setup.py clean;\
	rm -f *.png *.ps *.eps *.svg *.jpg *.pdf
	find . -name "_tmp*.py" | xargs rm -f;\
	find . \( -name "*~" -o -name "*.pyc" \) | xargs rm -f;\
	find unit \( -name "*.png" -o -name "*.ps"  -o -name "*.pdf" -o -name "*.eps" \) | xargs rm -f
	find . \( -name "#*" -o -name ".#*" -o -name ".*~" -o -name "*~" \) | xargs rm -f


