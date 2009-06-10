PYDIR = C:/Python26
PYTHON = ${PYDIR}/python.exe
SRCDIR = ${PWD}
WINSRCDIR = `${PWD}/data/mingw_path.sh ${PWD}`
ZLIBVERSION = 1.2.3
PNGVERSION = 1.2.36
FREETYPEVERSION = 2.3.9
TCLTKVERSION = 8.5.7
MPLVERSION = 0.98.5.3

## You shouldn't need to configure past this point

CFLAGS = -Os


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
	wget http://prdownloads.sourceforge.net/libpng/libpng-${PNGVERSION}.tar.bz2
	wget http://prdownloads.sourceforge.net/freetype/freetype-2.3.9.tar.bz2
	wget http://prdownloads.sourceforge.net/tcl/tcl${TCLTKVERSION}-src.tar.gz
	wget http://prdownloads.sourceforge.net/tcl/tk${TCLTKVERSION}-src.tar.gz

zlib:
	rm -rf zlib-${ZLIBVERSION}
	tar xvfz zlib-${ZLIBVERSION}.tar.gz
	cd zlib-${ZLIBVERSION} &&\
	export CFLAGS=${CFLAGS} &&\
	./configure &&\
	make -j3

# for reasons not clear to me, part of png compilation was failing
# because it could not find zlib.h, even with the CFLAGS which point
# to it and even with tryting to pass --includedir to configure.  So I
# manually copy the zlib *.h files into the png dir - JDH
png: zlib
	rm -rf libpng-${PNGVERSION}
	tar xvfj libpng-${PNGVERSION}.tar.bz2	
	cd libpng-${PNGVERSION} &&\
	cp ${SRCDIR}/zlib-${ZLIBVERSION}/*.h . && \
	export CFLAGS="${CFLAGS} -I${SRCDIR}/zlib-${ZLIBVERSION}" &&\
	export LDFLAGS="-L${SRCDIR}/zlib-${ZLIBVERSION}" &&\
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

freetype_hide:
	rm -rf freetype-${FREETYPEVERSION}
	tar xvfj freetype-${FREETYPEVERSION}.tar.bz2
	cd freetype-${FREETYPEVERSION} &&\
	export CFLAGS=${CFLAGS} &&\
	./configure --disable-shared &&\
	cp builds/win32/w32-mingw32.mk config.mk &&\
	make -j3 &&\
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
	${PYTHON} setupwin.py build_ext -c mingw32 -I ${PY_INCLUDE} -L ${PY_LINKER} bdist_wininst 
	${PYTHON} setupwinegg.py build_ext -c mingw32 -I ${PY_INCLUDE} -L ${PY_LINKER} bdist_egg


inplace:
	#rm -rf matplotlib-${MPLVERSION}
	#tar xvzf matplotlib-${MPLVERSION}.tar.gz
	cd matplotlib-${MPLVERSION} &&\
	rm -rf build lib/matplotlib/*.pyd lib/matplotlib/*.pyc lib/matplotlib/backends/*.pyd lib/matplotlib/backends/*.pyc &&\
	cp ../data/setup*.* . &&\
	${PYTHON} setup.py build_ext -c mingw32 -I ${PY_INCLUDE} -L ${PY_LINKER}  --inplace 
	cd matplotlib-${MPLVERSION}/lib &&\
	${PYTHON} -c 'import matplotlib; matplotlib.use("Agg"); from pylab import *; print matplotlib.__file__; plot([1,2,3]); savefig("test.png")'

test:
	${PYTHON} -c 'import matplotlib; matplotlib.use("Agg"); from pylab import *; print matplotlib.__file__; plot([1,2,3]); savefig("test.png")'

all: fetch_deps dependencies installers
