include ../../Makefile.in.$(shell uname)

PROGNAME=svg_test
OUTNAME=$(PROGNAME)
PLATFORM=BeOS

CXXFLAGS= $(AGGCXXFLAGS) -I../../include \
-I/boot/home/config/include/expat \
-L/boot/home/config/lib \
-L../../src \
$(PIXFMT)

LIBS = $(AGGLIBS) -lroot -lbe -ltranslation -lexpat

SRC=\
../$(PROGNAME).cpp \
../agg_svg_parser.cpp \
../agg_svg_path_renderer.cpp \
../agg_svg_path_tokenizer.cpp \
../../src/platform/$(PLATFORM)/agg_platform_support.cpp

OBJ= $(SRC:.cpp=.o)

$(OUTNAME):	$(OBJ)
	$(CXX) $(CXXFLAGS)  $^ -o $(OUTNAME) $(LIBS)
	mimeset -f $(OUTNAME)

clean:
	rm -f $(OUTNAME)
	rm -f ../$(PROGNAME).o
	rm -f ../../src/platform/$(PLATFORM)/agg_platform_support.o

%.o:	%.cpp
	@echo \< $*.cpp \>
	$(CXX) -c $(CXXFLAGS) $*.cpp -o $@


