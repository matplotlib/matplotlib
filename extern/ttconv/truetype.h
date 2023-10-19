/* -*- mode: c; c-basic-offset: 4 -*- */

/*
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */

#include <stdio.h>

/*
** ~ppr/src/include/typetype.h
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** This include file is shared by the source files
** "pprdrv/pprdrv_tt.c" and "pprdrv/pprdrv_tt2.c".
**
** Last modified 19 April 1995.
*/

/* Types used in TrueType font files. */
#define BYTE unsigned char
#define USHORT unsigned short int
#define SHORT short signed int
#define ULONG unsigned int
#define FIXED long signed int
#define FWord short signed int
#define uFWord short unsigned int

/* This structure stores a 16.16 bit fixed */
/* point number. */
typedef struct
    {
    short int whole;
    unsigned short int fraction;
    } Fixed;

/* This structure tells what we have found out about */
/* the current font. */
struct TTFONT
    {
    // A quick-and-dirty way to create a minimum level of exception safety
    // Added by Michael Droettboom
    TTFONT();
    ~TTFONT();

    const char *filename;               /* Name of TT file */
    FILE *file;                         /* the open TT file */
    font_type_enum  target_type;        /* 42 or 3 for PS, or -3 for PDF */

    ULONG numTables;                    /* number of tables present */
    char *PostName;                     /* Font's PostScript name */
    char *FullName;                     /* Font's full name */
    char *FamilyName;                   /* Font's family name */
    char *Style;                        /* Font's style string */
    char *Copyright;                    /* Font's copyright string */
    char *Version;                      /* Font's version string */
    char *Trademark;                    /* Font's trademark string */
    int llx,lly,urx,ury;                /* bounding box */

    Fixed TTVersion;                    /* Truetype version number from offset table */
    Fixed MfrRevision;                  /* Revision number of this font */

    BYTE *offset_table;                 /* Offset table in memory */
    BYTE *post_table;                   /* 'post' table in memory */

    BYTE *loca_table;                   /* 'loca' table in memory */
    BYTE *glyf_table;                   /* 'glyf' table in memory */
    BYTE *hmtx_table;                   /* 'hmtx' table in memory */

    USHORT numberOfHMetrics;
    int unitsPerEm;                     /* unitsPerEm converted to int */
    int HUPM;                           /* half of above */

    int numGlyphs;                      /* from 'post' table */

    int indexToLocFormat;               /* short or long offsets */
};

ULONG getULONG(BYTE *p);
USHORT getUSHORT(BYTE *p);
Fixed getFixed(BYTE *p);

/*
** Get an funits word.
** since it is 16 bits long, we can
** use getUSHORT() to do the real work.
*/
#define getFWord(x) (FWord)getUSHORT(x)
#define getuFWord(x) (uFWord)getUSHORT(x)

/*
** We can get a SHORT by making USHORT signed.
*/
#define getSHORT(x) (SHORT)getUSHORT(x)

/* This is the one routine in pprdrv_tt.c that is */
/* called from pprdrv_tt.c. */
const char *ttfont_CharStrings_getname(struct TTFONT *font, int charindex);

void tt_type3_charproc(TTStreamWriter& stream, struct TTFONT *font, int charindex);

/* Added 06-07-07 Michael Droettboom */
void ttfont_add_glyph_dependencies(struct TTFONT *font, std::vector<int>& glypy_ids);

/* This routine converts a number in the font's character coordinate */
/* system to a number in a 1000 unit character system. */
#define topost(x) (int)( ((int)(x) * 1000 + font->HUPM) / font->unitsPerEm )
#define topost2(x) (int)( ((int)(x) * 1000 + font.HUPM) / font.unitsPerEm )

/* Composite glyph values. */
#define ARG_1_AND_2_ARE_WORDS 1
#define ARGS_ARE_XY_VALUES 2
#define ROUND_XY_TO_GRID 4
#define WE_HAVE_A_SCALE 8
/* RESERVED 16 */
#define MORE_COMPONENTS 32
#define WE_HAVE_AN_X_AND_Y_SCALE 64
#define WE_HAVE_A_TWO_BY_TWO 128
#define WE_HAVE_INSTRUCTIONS 256
#define USE_MY_METRICS 512

/* end of file */
