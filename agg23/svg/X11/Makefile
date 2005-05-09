include ../../Makefile.in.$(shell uname)

PROGNAME=svg_test
OUTNAME=$(PROGNAME)
PLATFORM=X11

CXXFLAGS= $(AGGCXXFLAGS) -I../../include \
-L/usr/local/lib \
-L../../src \
$(PIXFMT)

LIBS = $(AGGLIBS) -lm -lX11 -lexpat

SRC=\
../$(PROGNAME).cpp \
../agg_svg_parser.cpp \
../agg_svg_path_renderer.cpp \
../agg_svg_path_tokenizer.cpp \
../../src/platform/$(PLATFORM)/agg_platform_support.cpp

OBJ=$(SRC:.cpp=.o)

$(OUTNAME):	$(OBJ)
	$(CXX) $(CXXFLAGS)  $^ -o $(OUTNAME) $(LIBS)

clean:
	rm -f $(OUTNAME)
	rm -f ../agg_svg_parser.o
	rm -f ../agg_svg_path_renderer.o
	rm -f ../agg_svg_path_tokenizer.o
	rm -f ../$(PROGNAME).o
	rm -f ../../src/platform/$(PLATFORM)/agg_platform_support.o

%.o:	%.cpp
	@echo \< $*.cpp \>
	$(CXX) -c $(CXXFLAGS) $*.cpp -o $@


