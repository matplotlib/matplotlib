PYTHON=python
SRCDIR=${PWD}
ZLIBVERSION=1.2.3
PNGVERSION=1.2.33
FREETYPEVERSION=2.3.7
MPLVERSION=0.98.5.3

## You shouldn't need to configure past this point

CFLAGS="-Os -I${SRCDIR}/zlib-${ZLIBVERSION} -I${SRCDIR}/libpng-${PNGVERSION} -I${SRCDIR}/freetype-${FREETYPEVERSION}/include"

LDFLAGS="-L${SRCDIR}/zlib-${ZLIBVERSION} -L${SRCDIR}/libpng-${PNGVERSION} -L${SRCDIR}/freetype-${FREETYPEVERSION}"

clean:
	rm -rf zlib-${ZLIBVERSION}.tar.gz libpng-${PNGVERSION}.tar.bz2 \
	freetype-${FREETYPEVERSION}.tar.bz2 \
	zlib-${ZLIBVERSION} libpng-${PNGVERSION} freetype-${FREETYPEVERSION} \
	matplotlib-${MPLVERSION} *~

fetch_deps:
	wget http://www.zlib.net/zlib-${ZLIBVERSION}.tar.gz
	wget http://internap.dl.sourceforge.net/sourceforge/libpng/libpng-${PNGVERSION}.tar.bz2
	wget http://download.savannah.gnu.org/releases/freetype/freetype-${FREETYPEVERSION}.tar.bz2

zlib:
	tar xvfz zlib-${ZLIBVERSION}.tar.gz
	cd zlib-${ZLIBVERSION} &&\
	./configure &&\
	make -j3

png: zlib
	tar xvfj libpng-${PNGVERSION}.tar.bz2
	cd libpng-${PNGVERSION} &&\
	export CFLAGS=${CFLAGS} &&\
	export LDFLAGS=${LDFLAGS} &&\
	./configure --disable-shared &&\
	make -j3 &&\
	cp .libs/libpng.a .

freetype:
	tar xvfj freetype-${FREETYPEVERSION}.tar.bz2 &&\
	cd freetype-${FREETYPEVERSION} &&\
	export GNUMAKE=mingw32-make &&\
	./configure --disable-shared &&\
	cp builds/win32/w32-mingw32.mk config.mk &&\
	mingw32-make -j3 &&\
	cp objs/libfreetype.a .

dependencies: png freetype

installers:
	tar xvzf matplotlib-${MPLVERSION}.tar.gz &&\
	cd matplotlib-${MPLVERSION} &&\
	rm -rf build &&\
	cp ../data/setup.cfg . &&\
	export CFLAGS=${CFLAGS} &&\
	export LDFLAGS=${LDFLAGS} &&\
	${PYTHON} setup.py build -c mingw32 bdist_wininst &&\
	${PYTHON) setupegg.py build -c mingw32 bdist_egg

all: fetch_deps dependencies installers
