#
# This makefile can be used to build a Win32 application under Cygwin
#

include ../../Makefile.in.$(shell uname)

PROGNAME=svg_test
OUTNAME=../$(PROGNAME)
PLATFORM=win32

CXXFLAGS= $(AGGCXXFLAGS) -I../../include \
-I/usr/X11R6/include \
-L/usr/X11R6/lib \
-L../../src \
$(PIXFMT)

LIBS = $(AGGLIBS) -lexpat -lm -lgdi32

SRC=\
../$(PROGNAME).cpp \
../agg_svg_parser.cpp \
../agg_svg_path_renderer.cpp \
../agg_svg_path_tokenizer.cpp \
../../src/platform/$(PLATFORM)/agg_platform_support.cpp \
../../src/platform/$(PLATFORM)/agg_win32_bmp.cpp

OBJ= $(SRC:.cpp=.o)

$(OUTNAME):	$(OBJ)
	$(CXX) $(CXXFLAGS)  $^ -o $(OUTNAME) $(LIBS)

clean:
	rm -f $(OUTNAME)
	rm -f ../*.o

%.o:	%.cpp
	@echo \< $*.cpp \>
	$(CXX) -c $(CXXFLAGS) $*.cpp -o $@


