# autogen.sh
#
# invoke the auto* tools to create the configureation system

# build aclocal.m4
aclocal

# build the configure script
autoconf

# set up libtool
libtoolize --force

# invoke automake
automake --foreign --add-missing

# and finally invoke our new configure
./configure $*

# end
