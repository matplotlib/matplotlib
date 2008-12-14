SRCDIR=${PWD}
ZLIBVERSION=1.2.3
PNGVERSION=1.2.33
FREETYPEVERSION=2.3.7
MPLVERSION=0.98.5
MPLSRC=matplotlib-0.98.5
MACOSX_DEPLOYMENT_TARGET=10.4

## You shouldn't need to configure past this point
CFLAGS="-Os -arch ppc -arch i386 -I${SRCDIR}/zlib-1.2.3 -I${SRCDIR}/libpng-1.2.33 -I${SRCDIR}/freetype-2.3.7/include"

LDFLAGS="-arch ppc -arch i386 -L${SRCDIR}/zlib-1.2.3 -L${SRCDIR}/libpng-1.2.33 -L${SRCDIR}/freetype-2.3.7"

CFLAGS_DEPS="-arch i386 -arch ppc -isysroot /Developer/SDKs/MacOSX10.4u.sdk"
LDFLAGS_DEPS="-arch i386 -arch ppc -syslibroot,/Developer/SDKs/MacOSX10.4u.sdk"

clean:
	rm -rf zlib-${ZLIBVERSION}.tar.gz libpng-${PNGVERSION}.tar.bz2 \
	freetype-${FREETYPEVERSION}.tar.bz2 bdist_mpkg-0.4.3.tar.gz \
	bdist_mpkg-0.4.3 \
	zlib-${ZLIBVERSION} libpng-${PNGVERSION} freetype-${FREETYPEVERSION} \
	matplotlib-${MPLVERSION} *~

fetch_deps:
	wget http://www.zlib.net/zlib-${ZLIBVERSION}.tar.gz &&\
	wget http://internap.dl.sourceforge.net/sourceforge/libpng/libpng-${PNGVERSION}.tar.bz2 &&\
	wget http://download.savannah.gnu.org/releases/freetype/freetype-${FREETYPEVERSION}.tar.bz2&&\
	wget http://pypi.python.org/packages/source/b/bdist_mpkg/bdist_mpkg-0.4.3.tar.gz&&\
	tar xvfz bdist_mpkg-0.4.3.tar.gz &&\
	patch -p0 < data/bdist.patch
	echo "You need to to install bdist_mpkg-0.4.3 now"



zlib:
	rm -rf zlib-${ZLIBVERSION} &&\
	tar xvfz zlib-${ZLIBVERSION}.tar.gz &&\
	cd zlib-${ZLIBVERSION} &&\
	./configure &&\
	MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} CFLAGS=${CFLAGS_DEPS} LDFLAGS=${LDFLAGS_DEPS} make -j3

png: zlib
	rm -rf libpng-${PNGVERSION} &&\
	tar xvfj libpng-${PNGVERSION}.tar.bz2
	cd libpng-${PNGVERSION} &&\
	./configure  --disable-dependency-tracking &&\
	MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} CFLAGS=${CFLAGS_DEPS} LDFLAGS=${LDFLAGS_DEPS} make -j3 &&\
	cp .libs/libpng.a .

freetype: zlib
	rm -rf ${FREETYPEVERSION} &&\
	tar xvfj freetype-${FREETYPEVERSION}.tar.bz2 &&\
	cd freetype-${FREETYPEVERSION} &&\
	./configure &&\
	MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET} CFLAGS=${CFLAGS_DEPS} LDFLAGS=${LDFLAGS_DEPS} make -j3 &&\
	cp objs/.libs/libfreetype.a .

dependencies:
	make zlib png freetype

installers:
	tar xvfz matplotlib-${MPLVERSION}.tar.gz && \
	cd ${MPLSRC} && \
	rm -rf build && \
	cp ../data/setup.cfg . &&\
	CFLAGS=${CFLAGS} LDFLAGS=${LDFLAGS} bdist_mpkg &&\
	CFLAGS=${CFLAGS} LDFLAGS=${LDFLAGS} python setupegg.py bdist_egg


