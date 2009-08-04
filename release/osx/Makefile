PYVERSION=2.6
PYTHON=python${PYVERSION}
SRCDIR=${PWD}
ZLIBVERSION=1.2.3
PNGVERSION=1.2.33
FREETYPEVERSION=2.3.7
MPLVERSION=0.99.0.rc1
BDISTMPKGVERSION=0.4.4
MPLSRC=matplotlib-${MPLVERSION}
MACOSX_DEPLOYMENT_TARGET=10.4

## You shouldn't need to configure past this point

CFLAGS="-Os -arch ppc -arch i386 -I${SRCDIR}/zlib-${ZLIBVERSION} -I${SRCDIR}/libpng-${PNGVERSION} -I${SRCDIR}/freetype-${FREETYPEVERSION}/include"

LDFLAGS="-arch ppc -arch i386 -L${SRCDIR}/zlib-${ZLIBVERSION} -L${SRCDIR}/libpng-${PNGVERSION} -L${SRCDIR}/freetype-${FREETYPEVERSION}"

CFLAGS_ZLIB="-arch i386 -arch ppc -isysroot /Developer/SDKs/MacOSX10.4u.sdk"
LDFLAGS_ZLIB="-arch i386 -arch ppc -syslibroot,/Developer/SDKs/MacOSX10.4u.sdk"

CFLAGS_DEPS="-arch i386 -arch ppc -I${SRCDIR}/zlib-${ZLIBVERSION} -isysroot /Developer/SDKs/MacOSX10.4u.sdk"
LDFLAGS_DEPS="-arch i386 -arch ppc -L${SRCDIR}/zlib-${ZLIBVERSION} -syslibroot,/Developer/SDKs/MacOSX10.4u.sdk"

clean:
	rm -rf zlib-${ZLIBVERSION}.tar.gz libpng-${PNGVERSION}.tar.bz2 \
	freetype-${FREETYPEVERSION}.tar.bz2 bdist_mpkg-${BDISTMPKGVERSION}.tar.gz \
	bdist_mpkg-${BDISTMPKGVERSION} \
	zlib-${ZLIBVERSION} libpng-${PNGVERSION} freetype-${FREETYPEVERSION} \
	matplotlib-${MPLVERSION} *~

fetch_deps:
	wget http://www.zlib.net/zlib-${ZLIBVERSION}.tar.gz &&\
	wget http://internap.dl.sourceforge.net/sourceforge/libpng/libpng-${PNGVERSION}.tar.bz2 &&\
	wget http://download.savannah.gnu.org/releases/freetype/freetype-${FREETYPEVERSION}.tar.bz2&&\
	wget http://pypi.python.org/packages/source/b/bdist_mpkg/bdist_mpkg-${BDISTMPKGVERSION}.tar.gz&&\
	tar xvfz bdist_mpkg-${BDISTMPKGVERSION}.tar.gz &&\
	echo "You need to to install bdist_mpkg-${BDISTMPKGVERSION} now"



zlib:
	unset PKG_CONFIG_PATH &&\
	rm -rf zlib-${ZLIBVERSION} &&\
	tar xvfz zlib-${ZLIBVERSION}.tar.gz &&\
	cd zlib-${ZLIBVERSION} &&\
	export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} &&\
	export CFLAGS=${CFLAGS_DEPS} &&\
	export LDFLAGS=${LDFLAGS_DEPS} &&\
	./configure &&\
	MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} CFLAGS=${CFLAGS_ZLIB} LDFLAGS=${LDFLAGS_ZLIB} make -j3&& \
	unset MACOSX_DEPLOYMENT_TARGET

png: zlib
	unset PKG_CONFIG_PATH &&\
	rm -rf libpng-${PNGVERSION} &&\
	tar xvfj libpng-${PNGVERSION}.tar.bz2
	cd libpng-${PNGVERSION} &&\
	export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} &&\
	export CFLAGS=${CFLAGS_DEPS} &&\
	export LDFLAGS=${LDFLAGS_DEPS} &&\
	./configure  --disable-dependency-tracking &&\
	make -j3 &&\
	cp .libs/libpng.a . &&\
	unset MACOSX_DEPLOYMENT_TARGET


freetype: zlib
	unset PKG_CONFIG_PATH &&\
	rm -rf ${FREETYPEVERSION} &&\
	tar xvfj freetype-${FREETYPEVERSION}.tar.bz2 &&\
	cd freetype-${FREETYPEVERSION} &&\
	export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} &&\
	export CFLAGS=${CFLAGS_DEPS} &&\
	export LDFLAGS=${LDFLAGS_DEPS} &&\
	./configure &&\
	make -j3 &&\
	cp objs/.libs/libfreetype.a .  &&\
	unset MACOSX_DEPLOYMENT_TARGET

dependencies:
	make zlib png freetype

installers:
	unset PKG_CONFIG_PATH &&\
	tar xvfz matplotlib-${MPLVERSION}.tar.gz && \
	cd ${MPLSRC} && \
	rm -rf build && \
	cp ../data/setup.cfg . &&\
	export CFLAGS=${CFLAGS} &&\
	export LDFLAGS=${LDFLAGS} &&\
	bdist_mpkg &&\
	${PYTHON} setupegg.py bdist_egg &&\
	cd dist && \
	zip -ro matplotlib-${MPLVERSION}-py${PYVERSION}-macosx10.5_mpkg.zip matplotlib-${MPLVERSION}-py${PYVERSION}-macosx10.5.mpkg

upload:
	rm -rf upload &&\
	mkdir upload &&\
	cp matplotlib-${MPLVERSION}.tar.gz upload/ &&\
	cp matplotlib-${MPLVERSION}/dist/matplotlib-${MPLVERSION}_r0-py${PYVERSION}-macosx-10.3-fat.egg upload/matplotlib-${MPLVERSION}-macosx-py${PYVERSION}.egg &&\
	cp matplotlib-${MPLVERSION}/dist/matplotlib-${MPLVERSION}-py${PYVERSION}-macosx10.5.zip upload/matplotlib-${MPLVERSION}-py${PYVERSION}-mpkg.zip&&\
	scp upload/* jdh2358@frs.sourceforge.net:uploads/

all:
	make clean fetch_deps dependencies installers upload



