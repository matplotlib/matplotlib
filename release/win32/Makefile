PYTHON = C:/Python26/python.exe
SRCDIR = ${PWD}# autoconf needs this path
WINSRCDIR = `${PWD}/data/mingw_path.sh ${PWD}`# distutils needs windows paths
ZLIBVERSION = 1.2.3
PNGVERSION = 1.2.33
FREETYPEVERSION = 2.3.7
#TCLTKVERSION = 8.4.19# Before Python 2.6
TCLTKVERSION = 8.5.7# Python 2.6
MPLVERSION = 0.98.6svn

## You shouldn't need to configure past this point

CFLAGS  = -Os
CFLAGS += -I${SRCDIR}/zlib-${ZLIBVERSION}
CFLAGS += -I${SRCDIR}/libpng-${PNGVERSION}
CFLAGS += -I${SRCDIR}/freetype-${FREETYPEVERSION}/include
CFLAGS += -I${SRCDIR}/tcl${TCLTKVERSION}-src/generic
CFLAGS += -I${SRCDIR}/tcl${TCLTKVERSION}-src/win
CFLAGS += -I${SRCDIR}/tk${TCLTKVERSION}-src/generic
CFLAGS += -I${SRCDIR}/tk${TCLTKVERSION}-src/win
CFLAGS += -I${SRCDIR}/tk${TCLTKVERSION}-src/X11

LDFLAGS  = -L${SRCDIR}/zlib-${ZLIBVERSION}
LDFLAGS += -L${SRCDIR}/libpng-${PNGVERSION}
LDFLAGS += -L${SRCDIR}/freetype-${FREETYPEVERSION}s

PY_INCLUDE = "${WINSRCDIR}\\zlib-${ZLIBVERSION};${WINSRCDIR}/libpng-${PNGVERSION};${WINSRCDIR}/freetype-${FREETYPEVERSION}/include;${WINSRCDIR}/tcl${TCLTKVERSION}/generic;${WINSRCDIR}/tcl${TCLTKVERSION}/win;${WINSRCDIR}/tk${TCLTKVERSION}/generic;${WINSRCDIR}/tk${TCLTKVERSION}/win;${WINSRCDIR}/tk${TCLTKVERSION}/xlib"

PY_LINKER = "${WINSRCDIR}/zlib-${ZLIBVERSION};${WINSRCDIR}/libpng-${PNGVERSION};${WINSRCDIR}/freetype-${FREETYPEVERSION}"

clean:
	rm -rf zlib-${ZLIBVERSION}.tar.gz libpng-${PNGVERSION}.tar.bz2 \
	freetype-${FREETYPEVERSION}.tar.bz2 \
	tcl${TCLTKVERSION}-src.tar.gz tk${TCLTKVERSION}-src.tar.gz \
	zlib-${ZLIBVERSION} libpng-${PNGVERSION} freetype-${FREETYPEVERSION} \
	tcl${TCLTKVERSION} tk${TCLTKVERSION} \
	matplotlib-${MPLVERSION} *~

fetch_deps:
	wget http://www.zlib.net/zlib-${ZLIBVERSION}.tar.gz
	wget http://internap.dl.sourceforge.net/sourceforge/libpng/libpng-${PNGVERSION}.tar.bz2
	wget http://download.savannah.gnu.org/releases/freetype/freetype-${FREETYPEVERSION}.tar.bz2
	wget http://prdownloads.sourceforge.net/tcl/tcl${TCLTKVERSION}-src.tar.gz
	wget http://prdownloads.sourceforge.net/tcl/tk${TCLTKVERSION}-src.tar.gz

zlib:
	rm -rf zlib-${ZLIBVERSION}
	tar xvfz zlib-${ZLIBVERSION}.tar.gz
	cd zlib-${ZLIBVERSION} &&\
	./configure &&\
	make -j3

png: zlib
	rm -rf libpng-${PNGVERSION}
	tar xvfj libpng-${PNGVERSION}.tar.bz2
	cd libpng-${PNGVERSION} &&\
	export CFLAGS="${CFLAGS}" &&\
	export LDFLAGS="${LDFLAGS}" &&\
	./configure --disable-shared &&\
	make -j3 &&\
	cp .libs/libpng.a .

freetype:
	rm -rf freetype-${FREETYPEVERSION}
	tar xvfj freetype-${FREETYPEVERSION}.tar.bz2
	cd freetype-${FREETYPEVERSION} &&\
	GNUMAKE=mingw32-make ./configure --disable-shared &&\
	cp builds/win32/w32-mingw32.mk config.mk &&\
	mingw32-make -j3 &&\
	cp objs/libfreetype.a .

tcltk:
	rm -rf tcl${TCLTKVERSION}
	rm -rf tk${TCLTKVERSION}
	tar xvfz tcl${TCLTKVERSION}-src.tar.gz
	tar xvfz tk${TCLTKVERSION}-src.tar.gz

dependencies: png freetype tcltk

installers:
	rm -rf matplotlib-${MPLVERSION}
	tar xvzf matplotlib-${MPLVERSION}.tar.gz
	cd matplotlib-${MPLVERSION} &&\
	rm -rf build &&\
	cp ../data/setup*.* . &&\
	${PYTHON} setupwin.py build_ext -c mingw32 -I ${PY_INCLUDE} -L ${PY_LINKER} bdist_wininst &&\
	${PYTHON} setupwinegg.py build_ext -c mingw32 -I ${PY_INCLUDE} -L ${PY_LINKER} bdist_egg

all: fetch_deps dependencies installers
