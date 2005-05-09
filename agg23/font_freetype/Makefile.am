INCLUDES = -I$(top_srcdir)/include `freetype-config --cflags`

noinst_LTLIBRARIES = libaggfontfreetype.la

libaggfontfreetype_la_LDFLAGS = -version-info 2:1:0 `freetype-config --libs`
libaggfontfreetype_la_SOURCES = agg_font_freetype.cpp
