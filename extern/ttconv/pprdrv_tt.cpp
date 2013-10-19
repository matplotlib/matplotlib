/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */

/*
** ~ppr/src/pprdrv/pprdrv_tt.c
** Copyright 1995, Trinity College Computing Center.
** Written by David Chappell.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** TrueType font support.  These functions allow PPR to generate
** PostScript fonts from Microsoft compatible TrueType font files.
**
** Last revised 19 December 1995.
*/

#include "global_defines.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pprdrv.h"
#include "truetype.h"
#include <sstream>
#include <Python.h>

/*==========================================================================
** Convert the indicated Truetype font file to a type 42 or type 3
** PostScript font and insert it in the output stream.
**
** All the routines from here to the end of file file are involved
** in this process.
==========================================================================*/

/*---------------------------------------
** Endian conversion routines.
** These routines take a BYTE pointer
** and return a value formed by reading
** bytes starting at that point.
**
** These routines read the big-endian
** values which are used in TrueType
** font files.
---------------------------------------*/

/*
** Get an Unsigned 32 bit number.
*/
ULONG getULONG(BYTE *p)
{
    int x;
    ULONG val=0;

    for (x=0; x<4; x++)
    {
        val *= 0x100;
        val += p[x];
    }

    return val;
} /* end of ftohULONG() */

/*
** Get an unsigned 16 bit number.
*/
USHORT getUSHORT(BYTE *p)
{
    int x;
    USHORT val=0;

    for (x=0; x<2; x++)
    {
        val *= 0x100;
        val += p[x];
    }

    return val;
} /* end of getUSHORT() */

/*
** Get a 32 bit fixed point (16.16) number.
** A special structure is used to return the value.
*/
Fixed getFixed(BYTE *s)
{
    Fixed val={0,0};

    val.whole = ((s[0] * 256) + s[1]);
    val.fraction = ((s[2] * 256) + s[3]);

    return val;
} /* end of getFixed() */

/*-----------------------------------------------------------------------
** Load a TrueType font table into memory and return a pointer to it.
** The font's "file" and "offset_table" fields must be set before this
** routine is called.
**
** This first argument is a TrueType font structure, the second
** argument is the name of the table to retrieve.  A table name
** is always 4 characters, though the last characters may be
** padding spaces.
-----------------------------------------------------------------------*/
BYTE *GetTable(struct TTFONT *font, const char *name)
{
    BYTE *ptr;
    ULONG x;

#ifdef DEBUG_TRUETYPE
    debug("GetTable(file,font,\"%s\")",name);
#endif

    /* We must search the table directory. */
    ptr = font->offset_table + 12;
    x=0;
    while (TRUE)
    {
        if ( strncmp((const char*)ptr,name,4) == 0 )
        {
            ULONG offset,length;
            BYTE *table;

            offset = getULONG( ptr + 8 );
            length = getULONG( ptr + 12 );
            table = (BYTE*)calloc( sizeof(BYTE), length );

            try
            {
#ifdef DEBUG_TRUETYPE
                debug("Loading table \"%s\" from offset %d, %d bytes",name,offset,length);
#endif

                if ( fseek( font->file, (long)offset, SEEK_SET ) )
                {
                    throw TTException("TrueType font may be corrupt (reason 3)");
                }

                if ( fread(table,sizeof(BYTE),length,font->file) != (sizeof(BYTE) * length))
                {
                    throw TTException("TrueType font may be corrupt (reason 4)");
                }
            }
            catch (TTException& )
            {
                free(table);
                throw;
            }
            return table;
        }

        x++;
        ptr += 16;
        if (x == font->numTables)
        {
            throw TTException("TrueType font is missing table");
        }
    }

} /* end of GetTable() */

static void utf16be_to_ascii(char *dst, char *src, size_t length) {
    ++src;
    for (; *src != 0 && length; dst++, src += 2, --length) {
        *dst = *src;
    }
}

/*--------------------------------------------------------------------
** Load the 'name' table, get information from it,
** and store that information in the font structure.
**
** The 'name' table contains information such as the name of
** the font, and it's PostScript name.
--------------------------------------------------------------------*/
void Read_name(struct TTFONT *font)
{
    BYTE *table_ptr,*ptr2;
    int numrecords;                     /* Number of strings in this table */
    BYTE *strings;                      /* pointer to start of string storage */
    int x;
    int platform;                       /* Current platform id */
    int nameid;                         /* name id, */
    int offset,length;                  /* offset and length of string. */

#ifdef DEBUG_TRUETYPE
    debug("Read_name()");
#endif

    table_ptr = NULL;

    /* Set default values to avoid future references to undefined
     * pointers. Allocate each of PostName, FullName, FamilyName,
     * Version, and Style separately so they can be freed safely. */
    for (char **ptr = &(font->PostName); ptr != NULL; )
    {
        *ptr = (char*) calloc(sizeof(char), strlen("unknown")+1);
        strcpy(*ptr, "unknown");
        if (ptr == &(font->PostName)) ptr = &(font->FullName);
        else if (ptr == &(font->FullName)) ptr = &(font->FamilyName);
        else if (ptr == &(font->FamilyName)) ptr = &(font->Version);
        else if (ptr == &(font->Version)) ptr = &(font->Style);
        else ptr = NULL;
    }
    font->Copyright = font->Trademark = (char*)NULL;

    table_ptr = GetTable(font, "name");         /* pointer to table */
    try
    {
        numrecords = getUSHORT( table_ptr + 2 );  /* number of names */
        strings = table_ptr + getUSHORT( table_ptr + 4 ); /* start of string storage */

        ptr2 = table_ptr + 6;
        for (x=0; x < numrecords; x++,ptr2+=12)
        {
            platform = getUSHORT(ptr2);
            nameid = getUSHORT(ptr2+6);
            length = getUSHORT(ptr2+8);
            offset = getUSHORT(ptr2+10);

#ifdef DEBUG_TRUETYPE
            debug("platform %d, encoding %d, language 0x%x, name %d, offset %d, length %d",
                  platform,encoding,language,nameid,offset,length);
#endif

            /* Copyright notice */
            if ( platform == 1 && nameid == 0 )
            {
                font->Copyright = (char*)calloc(sizeof(char),length+1);
                strncpy(font->Copyright,(const char*)strings+offset,length);
                font->Copyright[length]=(char)NULL;
                replace_newlines_with_spaces(font->Copyright);

#ifdef DEBUG_TRUETYPE
                debug("font->Copyright=\"%s\"",font->Copyright);
#endif
                continue;
            }


            /* Font Family name */
            if ( platform == 1 && nameid == 1 )
            {
                free(font->FamilyName);
                font->FamilyName = (char*)calloc(sizeof(char),length+1);
                strncpy(font->FamilyName,(const char*)strings+offset,length);
                font->FamilyName[length]=(char)NULL;
                replace_newlines_with_spaces(font->FamilyName);

#ifdef DEBUG_TRUETYPE
                debug("font->FamilyName=\"%s\"",font->FamilyName);
#endif
                continue;
            }


            /* Font Family name */
            if ( platform == 1 && nameid == 2 )
            {
                free(font->Style);
                font->Style = (char*)calloc(sizeof(char),length+1);
                strncpy(font->Style,(const char*)strings+offset,length);
                font->Style[length]=(char)NULL;
                replace_newlines_with_spaces(font->Style);

#ifdef DEBUG_TRUETYPE
                debug("font->Style=\"%s\"",font->Style);
#endif
                continue;
            }


            /* Full Font name */
            if ( platform == 1 && nameid == 4 )
            {
                free(font->FullName);
                font->FullName = (char*)calloc(sizeof(char),length+1);
                strncpy(font->FullName,(const char*)strings+offset,length);
                font->FullName[length]=(char)NULL;
                replace_newlines_with_spaces(font->FullName);

#ifdef DEBUG_TRUETYPE
                debug("font->FullName=\"%s\"",font->FullName);
#endif
                continue;
            }


            /* Version string */
            if ( platform == 1 && nameid == 5 )
            {
                free(font->Version);
                font->Version = (char*)calloc(sizeof(char),length+1);
                strncpy(font->Version,(const char*)strings+offset,length);
                font->Version[length]=(char)NULL;
                replace_newlines_with_spaces(font->Version);

#ifdef DEBUG_TRUETYPE
                debug("font->Version=\"%s\"",font->Version);
#endif
                continue;
            }


            /* PostScript name */
            if ( platform == 1 && nameid == 6 )
            {
                free(font->PostName);
                font->PostName = (char*)calloc(sizeof(char),length+1);
                strncpy(font->PostName,(const char*)strings+offset,length);
                font->PostName[length]=(char)NULL;
                replace_newlines_with_spaces(font->PostName);

#ifdef DEBUG_TRUETYPE
                debug("font->PostName=\"%s\"",font->PostName);
#endif
                continue;
            }

            /* Microsoft-format PostScript name */
            if ( platform == 3 && nameid == 6 )
            {
                free(font->PostName);
                font->PostName = (char*)calloc(sizeof(char),length+1);
                utf16be_to_ascii(font->PostName, (char *)strings+offset, length);
                font->PostName[length/2]=(char)NULL;
                replace_newlines_with_spaces(font->PostName);

#ifdef DEBUG_TRUETYPE
                debug("font->PostName=\"%s\"",font->PostName);
#endif
                continue;
            }


            /* Trademark string */
            if ( platform == 1 && nameid == 7 )
            {
                font->Trademark = (char*)calloc(sizeof(char),length+1);
                strncpy(font->Trademark,(const char*)strings+offset,length);
                font->Trademark[length]=(char)NULL;
                replace_newlines_with_spaces(font->Trademark);

#ifdef DEBUG_TRUETYPE
                debug("font->Trademark=\"%s\"",font->Trademark);
#endif
                continue;
            }
        }
    }
    catch (TTException& )
    {
        free(table_ptr);
        throw;
    }

    free(table_ptr);
} /* end of Read_name() */

/*---------------------------------------------------------------------
** Write the header for a PostScript font.
---------------------------------------------------------------------*/
void ttfont_header(TTStreamWriter& stream, struct TTFONT *font)
{
    int VMMin;
    int VMMax;

    /*
    ** To show that it is a TrueType font in PostScript format,
    ** we will begin the file with a specific string.
    ** This string also indicates the version of the TrueType
    ** specification on which the font is based and the
    ** font manufacturer's revision number for the font.
    */
    if ( font->target_type == PS_TYPE_42 ||
            font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("%%!PS-TrueTypeFont-%d.%d-%d.%d\n",
                      font->TTVersion.whole, font->TTVersion.fraction,
                      font->MfrRevision.whole, font->MfrRevision.fraction);
    }

    /* If it is not a Type 42 font, we will use a different format. */
    else
    {
        stream.putline("%!PS-Adobe-3.0 Resource-Font");
    }       /* See RBIIp 641 */

    /* We will make the title the name of the font. */
    stream.printf("%%%%Title: %s\n",font->FullName);

    /* If there is a Copyright notice, put it here too. */
    if ( font->Copyright != (char*)NULL )
    {
        stream.printf("%%%%Copyright: %s\n",font->Copyright);
    }

    /* We created this file. */
    if ( font->target_type == PS_TYPE_42 )
    {
        stream.putline("%%Creator: Converted from TrueType to type 42 by PPR");
    }
    else if (font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.putline("%%Creator: Converted from TypeType to type 42/type 3 hybrid by PPR");
    }
    else
    {
        stream.putline("%%Creator: Converted from TrueType to type 3 by PPR");
    }

    /* If VM usage information is available, print it. */
    if ( font->target_type == PS_TYPE_42 || font->target_type == PS_TYPE_42_3_HYBRID)
    {
        VMMin = (int)getULONG( font->post_table + 16 );
        VMMax = (int)getULONG( font->post_table + 20 );
        if ( VMMin > 0 && VMMax > 0 )
            stream.printf("%%%%VMUsage: %d %d\n",VMMin,VMMax);
    }

    /* Start the dictionary which will eventually */
    /* become the font. */
    if (font->target_type == PS_TYPE_42)
    {
        stream.putline("15 dict begin");
    }
    else
    {
        stream.putline("25 dict begin");

        /* Type 3 fonts will need some subroutines here. */
        stream.putline("/_d{bind def}bind def");
        stream.putline("/_m{moveto}_d");
        stream.putline("/_l{lineto}_d");
        stream.putline("/_cl{closepath eofill}_d");
        stream.putline("/_c{curveto}_d");
        stream.putline("/_sc{7 -1 roll{setcachedevice}{pop pop pop pop pop pop}ifelse}_d");
        stream.putline("/_e{exec}_d");
    }

    stream.printf("/FontName /%s def\n",font->PostName);
    stream.putline("/PaintType 0 def");

    if (font->target_type == PS_TYPE_42 || font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.putline("/FontMatrix[1 0 0 1 0 0]def");
    }
    else
    {
        stream.putline("/FontMatrix[.001 0 0 .001 0 0]def");
    }

    stream.printf("/FontBBox[%d %d %d %d]def\n",font->llx-1,font->lly-1,font->urx,font->ury);
    if (font->target_type == PS_TYPE_42 || font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("/FontType 42 def\n", font->target_type );
    }
    else
    {
        stream.printf("/FontType 3 def\n", font->target_type );
    }
} /* end of ttfont_header() */

/*-------------------------------------------------------------
** Define the encoding array for this font.
** Since we don't really want to deal with converting all of
** the possible font encodings in the wild to a standard PS
** one, we just explicitly create one for each font.
-------------------------------------------------------------*/
void ttfont_encoding(TTStreamWriter& stream, struct TTFONT *font, std::vector<int>& glyph_ids, font_type_enum target_type)
{
    if (target_type == PS_TYPE_3 || target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("/Encoding [ ");

        for (std::vector<int>::const_iterator i = glyph_ids.begin();
                i != glyph_ids.end(); ++i)
        {
            const char* name = ttfont_CharStrings_getname(font, *i);
            stream.printf("/%s ", name);
        }

        stream.printf("] def\n");
    }
    else
    {
        stream.putline("/Encoding StandardEncoding def");
    }
} /* end of ttfont_encoding() */

/*-----------------------------------------------------------
** Create the optional "FontInfo" sub-dictionary.
-----------------------------------------------------------*/
void ttfont_FontInfo(TTStreamWriter& stream, struct TTFONT *font)
{
    Fixed ItalicAngle;

    /* We create a sub dictionary named "FontInfo" where we */
    /* store information which though it is not used by the */
    /* interpreter, is useful to some programs which will */
    /* be printing with the font. */
    stream.putline("/FontInfo 10 dict dup begin");

    /* These names come from the TrueType font's "name" table. */
    stream.printf("/FamilyName (%s) def\n",font->FamilyName);
    stream.printf("/FullName (%s) def\n",font->FullName);

    if ( font->Copyright != (char*)NULL || font->Trademark != (char*)NULL )
    {
        stream.printf("/Notice (%s",
                      font->Copyright != (char*)NULL ? font->Copyright : "");
        stream.printf("%s%s) def\n",
                      font->Trademark != (char*)NULL ? " " : "",
                      font->Trademark != (char*)NULL ? font->Trademark : "");
    }

    /* This information is not quite correct. */
    stream.printf("/Weight (%s) def\n",font->Style);

    /* Some fonts have this as "version". */
    stream.printf("/Version (%s) def\n",font->Version);

    /* Some information from the "post" table. */
    ItalicAngle = getFixed( font->post_table + 4 );
    stream.printf("/ItalicAngle %d.%d def\n",ItalicAngle.whole,ItalicAngle.fraction);
    stream.printf("/isFixedPitch %s def\n", getULONG( font->post_table + 12 ) ? "true" : "false" );
    stream.printf("/UnderlinePosition %d def\n", (int)getFWord( font->post_table + 8 ) );
    stream.printf("/UnderlineThickness %d def\n", (int)getFWord( font->post_table + 10 ) );
    stream.putline("end readonly def");
} /* end of ttfont_FontInfo() */

/*-------------------------------------------------------------------
** sfnts routines
** These routines generate the PostScript "sfnts" array which
** contains one or more strings which contain a reduced version
** of the TrueType font.
**
** A number of functions are required to accomplish this rather
** complicated task.
-------------------------------------------------------------------*/
int string_len;
int line_len;
int in_string;

/*
** This is called once at the start.
*/
void sfnts_start(TTStreamWriter& stream)
{
    stream.puts("/sfnts[<");
    in_string=TRUE;
    string_len=0;
    line_len=8;
} /* end of sfnts_start() */

/*
** Write a BYTE as a hexadecimal value as part of the sfnts array.
*/
void sfnts_pputBYTE(TTStreamWriter& stream, BYTE n)
{
    static const char hexdigits[]="0123456789ABCDEF";

    if (!in_string)
    {
        stream.put_char('<');
        string_len=0;
        line_len++;
        in_string=TRUE;
    }

    stream.put_char( hexdigits[ n / 16 ] );
    stream.put_char( hexdigits[ n % 16 ] );
    string_len++;
    line_len+=2;

    if (line_len > 70)
    {
        stream.put_char('\n');
        line_len=0;
    }

} /* end of sfnts_pputBYTE() */

/*
** Write a USHORT as a hexadecimal value as part of the sfnts array.
*/
void sfnts_pputUSHORT(TTStreamWriter& stream, USHORT n)
{
    sfnts_pputBYTE(stream, n / 256);
    sfnts_pputBYTE(stream, n % 256);
} /* end of sfnts_pputUSHORT() */

/*
** Write a ULONG as part of the sfnts array.
*/
void sfnts_pputULONG(TTStreamWriter& stream, ULONG n)
{
    int x1,x2,x3;

    x1 = n % 256;
    n /= 256;
    x2 = n % 256;
    n /= 256;
    x3 = n % 256;
    n /= 256;

    sfnts_pputBYTE(stream, n);
    sfnts_pputBYTE(stream, x3);
    sfnts_pputBYTE(stream, x2);
    sfnts_pputBYTE(stream, x1);
} /* end of sfnts_pputULONG() */

/*
** This is called whenever it is
** necessary to end a string in the sfnts array.
**
** (The array must be broken into strings which are
** no longer than 64K characters.)
*/
void sfnts_end_string(TTStreamWriter& stream)
{
    if (in_string)
    {
        string_len=0;           /* fool sfnts_pputBYTE() */

#ifdef DEBUG_TRUETYPE_INLINE
        puts("\n% dummy byte:\n");
#endif

        sfnts_pputBYTE(stream, 0);      /* extra byte for pre-2013 compatibility */
        stream.put_char('>');
        line_len++;
    }
    in_string=FALSE;
} /* end of sfnts_end_string() */

/*
** This is called at the start of each new table.
** The argement is the length in bytes of the table
** which will follow.  If the new table will not fit
** in the current string, a new one is started.
*/
void sfnts_new_table(TTStreamWriter& stream, ULONG length)
{
    if ( (string_len + length) > 65528 )
        sfnts_end_string(stream);
} /* end of sfnts_new_table() */

/*
** We may have to break up the 'glyf' table.  That is the reason
** why we provide this special routine to copy it into the sfnts
** array.
*/
void sfnts_glyf_table(TTStreamWriter& stream, struct TTFONT *font, ULONG oldoffset, ULONG correct_total_length)
{
    ULONG off;
    ULONG length;
    int c;
    ULONG total=0;              /* running total of bytes written to table */
    int x;
    bool loca_is_local=false;

#ifdef DEBUG_TRUETYPE
    debug("sfnts_glyf_table(font,%d)", (int)correct_total_length);
#endif

    if (font->loca_table == NULL)
    {
        font->loca_table = GetTable(font,"loca");
        loca_is_local = true;
    }

    /* Seek to proper position in the file. */
    fseek( font->file, oldoffset, SEEK_SET );

    /* Copy the glyphs one by one */
    for (x=0; x < font->numGlyphs; x++)
    {
        /* Read the glyph offset from the index-to-location table. */
        if (font->indexToLocFormat == 0)
        {
            off = getUSHORT( font->loca_table + (x * 2) );
            off *= 2;
            length = getUSHORT( font->loca_table + ((x+1) * 2) );
            length *= 2;
            length -= off;
        }
        else
        {
            off = getULONG( font->loca_table + (x * 4) );
            length = getULONG( font->loca_table + ((x+1) * 4) );
            length -= off;
        }

#ifdef DEBUG_TRUETYPE
        debug("glyph length=%d",(int)length);
#endif

        /* Start new string if necessary. */
        sfnts_new_table( stream, (int)length );

        /*
        ** Make sure the glyph is padded out to a
        ** two byte boundary.
        */
        if ( length % 2 ) {
            throw TTException("TrueType font contains a 'glyf' table without 2 byte padding");
        }

        /* Copy the bytes of the glyph. */
        while ( length-- )
        {
            if ( (c = fgetc(font->file)) == EOF ) {
                throw TTException("TrueType font may be corrupt (reason 6)");
            }

            sfnts_pputBYTE(stream, c);
            total++;            /* add to running total */
        }

    }

    if (loca_is_local)
    {
        free(font->loca_table);
        font->loca_table = NULL;
    }

    /* Pad out to full length from table directory */
    while ( total < correct_total_length )
    {
        sfnts_pputBYTE(stream, 0);
        total++;
    }

} /* end of sfnts_glyf_table() */

/*
** Here is the routine which ties it all together.
**
** Create the array called "sfnts" which
** holds the actual TrueType data.
*/
void ttfont_sfnts(TTStreamWriter& stream, struct TTFONT *font)
{
    static const char *table_names[] =  /* The names of all tables */
    {
        /* which it is worth while */
        "cvt ",                         /* to include in a Type 42 */
        "fpgm",                         /* PostScript font. */
        "glyf",
        "head",
        "hhea",
        "hmtx",
        "loca",
        "maxp",
        "prep"
    } ;

    struct                      /* The location of each of */
    {
        ULONG oldoffset;        /* the above tables. */
        ULONG newoffset;
        ULONG length;
        ULONG checksum;
    } tables[9];

    BYTE *ptr;                  /* A pointer into the origional table directory. */
    ULONG x,y;                  /* General use loop countes. */
    int c;                      /* Input character. */
    int diff;
    ULONG nextoffset;
    int count;                  /* How many `important' tables did we find? */

    ptr = font->offset_table + 12;
    nextoffset=0;
    count=0;

    /*
    ** Find the tables we want and store there vital
    ** statistics in tables[].
    */
    for (x=0; x < 9; x++ )
    {
        do
        {
            diff = strncmp( (char*)ptr, table_names[x], 4 );

            if ( diff > 0 )             /* If we are past it. */
            {
                tables[x].length = 0;
                diff = 0;
            }
            else if ( diff < 0 )        /* If we haven't hit it yet. */
            {
                ptr += 16;
            }
            else if ( diff == 0 )       /* Here it is! */
            {
                tables[x].newoffset = nextoffset;
                tables[x].checksum = getULONG( ptr + 4 );
                tables[x].oldoffset = getULONG( ptr + 8 );
                tables[x].length = getULONG( ptr + 12 );
                nextoffset += ( ((tables[x].length + 3) / 4) * 4 );
                count++;
                ptr += 16;
            }
        }
        while (diff != 0);

    } /* end of for loop which passes over the table directory */

    /* Begin the sfnts array. */
    sfnts_start(stream);

    /* Generate the offset table header */
    /* Start by copying the TrueType version number. */
    ptr = font->offset_table;
    for (x=0; x < 4; x++)
    {
        sfnts_pputBYTE( stream,  *(ptr++) );
    }

    /* Now, generate those silly numTables numbers. */
    sfnts_pputUSHORT(stream, count);            /* number of tables */
    if ( count == 9 )
    {
        sfnts_pputUSHORT(stream, 7);          /* searchRange */
        sfnts_pputUSHORT(stream, 3);          /* entrySelector */
        sfnts_pputUSHORT(stream, 81);         /* rangeShift */
    }
#ifdef DEBUG_TRUETYPE
    else
    {
        debug("only %d tables selected",count);
    }
#endif

    /* Now, emmit the table directory. */
    for (x=0; x < 9; x++)
    {
        if ( tables[x].length == 0 )    /* Skip missing tables */
        {
            continue;
        }

        /* Name */
        sfnts_pputBYTE( stream, table_names[x][0] );
        sfnts_pputBYTE( stream, table_names[x][1] );
        sfnts_pputBYTE( stream, table_names[x][2] );
        sfnts_pputBYTE( stream, table_names[x][3] );

        /* Checksum */
        sfnts_pputULONG( stream, tables[x].checksum );

        /* Offset */
        sfnts_pputULONG( stream, tables[x].newoffset + 12 + (count * 16) );

        /* Length */
        sfnts_pputULONG( stream, tables[x].length );
    }

    /* Now, send the tables */
    for (x=0; x < 9; x++)
    {
        if ( tables[x].length == 0 )    /* skip tables that aren't there */
        {
            continue;
        }

#ifdef DEBUG_TRUETYPE
        debug("emmiting table '%s'",table_names[x]);
#endif

        /* 'glyf' table gets special treatment */
        if ( strcmp(table_names[x],"glyf")==0 )
        {
            sfnts_glyf_table(stream,font,tables[x].oldoffset,tables[x].length);
        }
        else                    /* Other tables may not exceed */
        {
            /* 65535 bytes in length. */
            if ( tables[x].length > 65535 )
            {
                throw TTException("TrueType font has a table which is too long");
            }

            /* Start new string if necessary. */
            sfnts_new_table(stream, tables[x].length);

            /* Seek to proper position in the file. */
            fseek( font->file, tables[x].oldoffset, SEEK_SET );

            /* Copy the bytes of the table. */
            for ( y=0; y < tables[x].length; y++ )
            {
                if ( (c = fgetc(font->file)) == EOF )
                {
                    throw TTException("TrueType font may be corrupt (reason 7)");
                }

                sfnts_pputBYTE(stream, c);
            }
        }

        /* Padd it out to a four byte boundary. */
        y=tables[x].length;
        while ( (y % 4) != 0 )
        {
            sfnts_pputBYTE(stream, 0);
            y++;
#ifdef DEBUG_TRUETYPE_INLINE
            puts("\n% pad byte:\n");
#endif
        }

    } /* End of loop for all tables */

    /* Close the array. */
    sfnts_end_string(stream);
    stream.putline("]def");
} /* end of ttfont_sfnts() */

/*--------------------------------------------------------------
** Create the CharStrings dictionary which will translate
** PostScript character names to TrueType font character
** indexes.
**
** If we are creating a type 3 instead of a type 42 font,
** this array will instead convert PostScript character names
** to executable proceedures.
--------------------------------------------------------------*/
const char *Apple_CharStrings[]=
{
    ".notdef",".null","nonmarkingreturn","space","exclam","quotedbl","numbersign",
    "dollar","percent","ampersand","quotesingle","parenleft","parenright",
    "asterisk","plus", "comma","hyphen","period","slash","zero","one","two",
    "three","four","five","six","seven","eight","nine","colon","semicolon",
    "less","equal","greater","question","at","A","B","C","D","E","F","G","H","I",
    "J","K", "L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "bracketleft","backslash","bracketright","asciicircum","underscore","grave",
    "a","b","c","d","e","f","g","h","i","j","k", "l","m","n","o","p","q","r","s",
    "t","u","v","w","x","y","z","braceleft","bar","braceright","asciitilde",
    "Adieresis","Aring","Ccedilla","Eacute","Ntilde","Odieresis","Udieresis",
    "aacute","agrave","acircumflex","adieresis","atilde","aring","ccedilla",
    "eacute","egrave","ecircumflex","edieresis","iacute","igrave","icircumflex",
    "idieresis","ntilde","oacute","ograve","ocircumflex","odieresis","otilde",
    "uacute","ugrave","ucircumflex","udieresis","dagger","degree","cent",
    "sterling","section","bullet","paragraph","germandbls","registered",
    "copyright","trademark","acute","dieresis","notequal","AE","Oslash",
    "infinity","plusminus","lessequal","greaterequal","yen","mu","partialdiff",
    "summation","product","pi","integral","ordfeminine","ordmasculine","Omega",
    "ae","oslash","questiondown","exclamdown","logicalnot","radical","florin",
    "approxequal","Delta","guillemotleft","guillemotright","ellipsis",
    "nobreakspace","Agrave","Atilde","Otilde","OE","oe","endash","emdash",
    "quotedblleft","quotedblright","quoteleft","quoteright","divide","lozenge",
    "ydieresis","Ydieresis","fraction","currency","guilsinglleft","guilsinglright",
    "fi","fl","daggerdbl","periodcentered","quotesinglbase","quotedblbase",
    "perthousand","Acircumflex","Ecircumflex","Aacute","Edieresis","Egrave",
    "Iacute","Icircumflex","Idieresis","Igrave","Oacute","Ocircumflex","apple",
    "Ograve","Uacute","Ucircumflex","Ugrave","dotlessi","circumflex","tilde",
    "macron","breve","dotaccent","ring","cedilla","hungarumlaut","ogonek","caron",
    "Lslash","lslash","Scaron","scaron","Zcaron","zcaron","brokenbar","Eth","eth",
    "Yacute","yacute","Thorn","thorn","minus","multiply","onesuperior",
    "twosuperior","threesuperior","onehalf","onequarter","threequarters","franc",
    "Gbreve","gbreve","Idot","Scedilla","scedilla","Cacute","cacute","Ccaron",
    "ccaron","dmacron","markingspace","capslock","shift","propeller","enter",
    "markingtabrtol","markingtabltor","control","markingdeleteltor",
    "markingdeletertol","option","escape","parbreakltor","parbreakrtol",
    "newpage","checkmark","linebreakltor","linebreakrtol","markingnobreakspace",
    "diamond","appleoutline"
};

/*
** This routine is called by the one below.
** It is also called from pprdrv_tt2.c
*/
const char *ttfont_CharStrings_getname(struct TTFONT *font, int charindex)
{
    int GlyphIndex;
    static char temp[80];
    char *ptr;
    ULONG len;

    Fixed post_format;

    /* The 'post' table format number. */
    post_format = getFixed( font->post_table );

    if ( post_format.whole != 2 || post_format.fraction != 0 )
    {
        /* We don't have a glyph name table, so generate a name.
           This generated name must match exactly the name that is
           generated by FT2Font in get_glyph_name */
        PyOS_snprintf(temp, 80, "uni%08x", charindex);
        return temp;
    }

    GlyphIndex = (int)getUSHORT( font->post_table + 34 + (charindex * 2) );

    if ( GlyphIndex <= 257 )            /* If a standard Apple name, */
    {
        return Apple_CharStrings[GlyphIndex];
    }
    else                                /* Otherwise, use one */
    {
        /* of the pascal strings. */
        GlyphIndex -= 258;

        /* Set pointer to start of Pascal strings. */
        ptr = (char*)(font->post_table + 34 + (font->numGlyphs * 2));

        len = (ULONG)*(ptr++);  /* Step thru the strings */
        while (GlyphIndex--)            /* until we get to the one */
        {
            /* that we want. */
            ptr += len;
            len = (ULONG)*(ptr++);
        }

        if ( len >= sizeof(temp) )
        {
            throw TTException("TrueType font file contains a very long PostScript name");
        }

        strncpy(temp,ptr,len);  /* Copy the pascal string into */
        temp[len]=(char)NULL;   /* a buffer and make it ASCIIz. */

        return temp;
    }
} /* end of ttfont_CharStrings_getname() */

/*
** This is the central routine of this section.
*/
void ttfont_CharStrings(TTStreamWriter& stream, struct TTFONT *font, std::vector<int>& glyph_ids)
{
    Fixed post_format;

    /* The 'post' table format number. */
    post_format = getFixed( font->post_table );

    /* Emmit the start of the PostScript code to define the dictionary. */
    stream.printf("/CharStrings %d dict dup begin\n", glyph_ids.size());

    /* Emmit one key-value pair for each glyph. */
    for (std::vector<int>::const_iterator i = glyph_ids.begin();
            i != glyph_ids.end(); ++i)
    {
        if ((font->target_type == PS_TYPE_42 ||
             font->target_type == PS_TYPE_42_3_HYBRID)
            && *i < 256) /* type 42 */
        {
            stream.printf("/%s %d def\n",ttfont_CharStrings_getname(font, *i), *i);
        }
        else                            /* type 3 */
        {
            stream.printf("/%s{",ttfont_CharStrings_getname(font, *i));

            tt_type3_charproc(stream, font, *i);

            stream.putline("}_d");      /* "} bind def" */
        }
    }

    stream.putline("end readonly def");
} /* end of ttfont_CharStrings() */

/*----------------------------------------------------------------
** Emmit the code to finish up the dictionary and turn
** it into a font.
----------------------------------------------------------------*/
void ttfont_trailer(TTStreamWriter& stream, struct TTFONT *font)
{
    /* If we are generating a type 3 font, we need to provide */
    /* a BuildGlyph and BuildChar proceedures. */
    if (font->target_type == PS_TYPE_3 ||
        font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.put_char('\n');

        stream.putline("/BuildGlyph");
        stream.putline(" {exch begin");         /* start font dictionary */
        stream.putline(" CharStrings exch");
        stream.putline(" 2 copy known not{pop /.notdef}if");
        stream.putline(" true 3 1 roll get exec");
        stream.putline(" end}_d");

        stream.put_char('\n');

        /* This proceedure is for compatiblity with */
        /* level 1 interpreters. */
        stream.putline("/BuildChar {");
        stream.putline(" 1 index /Encoding get exch get");
        stream.putline(" 1 index /BuildGlyph get exec");
        stream.putline("}_d");

        stream.put_char('\n');
    }

    /* If we are generating a type 42 font, we need to check to see */
    /* if this PostScript interpreter understands type 42 fonts.  If */
    /* it doesn't, we will hope that the Apple TrueType rasterizer */
    /* has been loaded and we will adjust the font accordingly. */
    /* I found out how to do this by examining a TrueType font */
    /* generated by a Macintosh.  That is where the TrueType interpreter */
    /* setup instructions and part of BuildGlyph came from. */
    if (font->target_type == PS_TYPE_42 ||
        font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.put_char('\n');

        /* If we have no "resourcestatus" command, or FontType 42 */
        /* is unknown, leave "true" on the stack. */
        stream.putline("systemdict/resourcestatus known");
        stream.putline(" {42 /FontType resourcestatus");
        stream.putline("   {pop pop false}{true}ifelse}");
        stream.putline(" {true}ifelse");

        /* If true, execute code to produce an error message if */
        /* we can't find Apple's TrueDict in VM. */
        stream.putline("{/TrueDict where{pop}{(%%[ Error: no TrueType rasterizer ]%%)= flush}ifelse");

        /* Since we are expected to use Apple's TrueDict TrueType */
        /* reasterizer, change the font type to 3. */
        stream.putline("/FontType 3 def");

        /* Define a string to hold the state of the Apple */
        /* TrueType interpreter. */
        stream.putline(" /TrueState 271 string def");

        /* It looks like we get information about the resolution */
        /* of the printer and store it in the TrueState string. */
        stream.putline(" TrueDict begin sfnts save");
        stream.putline(" 72 0 matrix defaultmatrix dtransform dup");
        stream.putline(" mul exch dup mul add sqrt cvi 0 72 matrix");
        stream.putline(" defaultmatrix dtransform dup mul exch dup");
        stream.putline(" mul add sqrt cvi 3 -1 roll restore");
        stream.putline(" TrueState initer end");

        /* This BuildGlyph procedure will look the name up in the */
        /* CharStrings array, and then check to see if what it gets */
        /* is a procedure.  If it is, it executes it, otherwise, it */
        /* lets the TrueType rasterizer loose on it. */

        /* When this proceedure is executed the stack contains */
        /* the font dictionary and the character name.  We */
        /* exchange arguments and move the dictionary to the */
        /* dictionary stack. */
        stream.putline(" /BuildGlyph{exch begin");
        /* stack: charname */

        /* Put two copies of CharStrings on the stack and consume */
        /* one testing to see if the charname is defined in it, */
        /* leave the answer on the stack. */
        stream.putline("  CharStrings dup 2 index known");
        /* stack: charname CharStrings bool */

        /* Exchange the CharStrings dictionary and the charname, */
        /* but if the answer was false, replace the character name */
        /* with ".notdef". */
        stream.putline("    {exch}{exch pop /.notdef}ifelse");
        /* stack: CharStrings charname */

        /* Get the value from the CharStrings dictionary and see */
        /* if it is executable. */
        stream.putline("  get dup xcheck");
        /* stack: CharStrings_entry */

        /* If is a proceedure.  Execute according to RBIIp 277-278. */
        stream.putline("    {currentdict systemdict begin begin exec end end}");

        /* Is a TrueType character index, let the rasterizer at it. */
        stream.putline("    {TrueDict begin /bander load cvlit exch TrueState render end}");

        stream.putline("    ifelse");

        /* Pop the font's dictionary off the stack. */
        stream.putline(" end}bind def");

        /* This is the level 1 compatibility BuildChar procedure. */
        /* See RBIIp 281. */
        stream.putline(" /BuildChar{");
        stream.putline("  1 index /Encoding get exch get");
        stream.putline("  1 index /BuildGlyph get exec");
        stream.putline(" }bind def");

        /* Here we close the condition which is true */
        /* if the printer has no built-in TrueType */
        /* rasterizer. */
        stream.putline("}if");
        stream.put_char('\n');
    } /* end of if Type 42 not understood. */

    stream.putline("FontName currentdict end definefont pop");
    /* stream.putline("%%EOF"); */
} /* end of ttfont_trailer() */

/*------------------------------------------------------------------
** This is the externally callable routine which inserts the font.
------------------------------------------------------------------*/

void read_font(const char *filename, font_type_enum target_type, std::vector<int>& glyph_ids, TTFONT& font)
{
    BYTE *ptr;

    /* Decide what type of PostScript font we will be generating. */
    font.target_type = target_type;

    if (font.target_type == PS_TYPE_42)
    {
        bool has_low = false;
        bool has_high = false;

        for (std::vector<int>::const_iterator i = glyph_ids.begin();
                i != glyph_ids.end(); ++i)
        {
            if (*i > 255)
            {
                has_high = true;
                if (has_low) break;
            }
            else
            {
                has_low = true;
                if (has_high) break;
            }
        }

        if (has_high && has_low)
        {
            font.target_type = PS_TYPE_42_3_HYBRID;
        }
        else if (has_high && !has_low)
        {
            font.target_type = PS_TYPE_3;
        }
    }

    /* Save the file name for error messages. */
    font.filename=filename;

    /* Open the font file */
    if ( (font.file = fopen(filename,"rb")) == (FILE*)NULL )
    {
        throw TTException("Failed to open TrueType font");
    }

    /* Allocate space for the unvarying part of the offset table. */
    assert(font.offset_table == NULL);
    font.offset_table = (BYTE*)calloc( 12, sizeof(BYTE) );

    /* Read the first part of the offset table. */
    if ( fread( font.offset_table, sizeof(BYTE), 12, font.file ) != 12 )
    {
        throw TTException("TrueType font may be corrupt (reason 1)");
    }

    /* Determine how many directory entries there are. */
    font.numTables = getUSHORT( font.offset_table + 4 );
#ifdef DEBUG_TRUETYPE
    debug("numTables=%d",(int)font.numTables);
#endif

    /* Expand the memory block to hold the whole thing. */
    font.offset_table = (BYTE*)realloc( font.offset_table, sizeof(BYTE) * (12 + font.numTables * 16) );

    /* Read the rest of the table directory. */
    if ( fread( font.offset_table + 12, sizeof(BYTE), (font.numTables*16), font.file ) != (font.numTables*16) )
    {
        throw TTException("TrueType font may be corrupt (reason 2)");
    }

    /* Extract information from the "Offset" table. */
    font.TTVersion = getFixed( font.offset_table );

    /* Load the "head" table and extract information from it. */
    ptr = GetTable(&font, "head");
    try
    {
        font.MfrRevision = getFixed( ptr + 4 );           /* font revision number */
        font.unitsPerEm = getUSHORT( ptr + 18 );
        font.HUPM = font.unitsPerEm / 2;
#ifdef DEBUG_TRUETYPE
        debug("unitsPerEm=%d",(int)font.unitsPerEm);
#endif
        font.llx = topost2( getFWord( ptr + 36 ) );               /* bounding box info */
        font.lly = topost2( getFWord( ptr + 38 ) );
        font.urx = topost2( getFWord( ptr + 40 ) );
        font.ury = topost2( getFWord( ptr + 42 ) );
        font.indexToLocFormat = getSHORT( ptr + 50 );     /* size of 'loca' data */
        if (font.indexToLocFormat != 0 && font.indexToLocFormat != 1)
        {
            throw TTException("TrueType font is unusable because indexToLocFormat != 0");
        }
        if ( getSHORT(ptr+52) != 0 )
        {
            throw TTException("TrueType font is unusable because glyphDataFormat != 0");
        }
    }
    catch (TTException& )
    {
        free(ptr);
        throw;
    }
    free(ptr);

    /* Load information from the "name" table. */
    Read_name(&font);

    /* We need to have the PostScript table around. */
    assert(font.post_table == NULL);
    font.post_table = GetTable(&font, "post");
    font.numGlyphs = getUSHORT( font.post_table + 32 );

    /* If we are generating a Type 3 font, we will need to */
    /* have the 'loca' and 'glyf' tables arround while */
    /* we are generating the CharStrings. */
    if (font.target_type == PS_TYPE_3 || font.target_type == PDF_TYPE_3 ||
            font.target_type == PS_TYPE_42_3_HYBRID)
    {
        BYTE *ptr;                      /* We need only one value */
        ptr = GetTable(&font, "hhea");
        font.numberOfHMetrics = getUSHORT(ptr + 34);
        free(ptr);

        assert(font.loca_table == NULL);
        font.loca_table = GetTable(&font,"loca");
        assert(font.glyf_table == NULL);
        font.glyf_table = GetTable(&font,"glyf");
        assert(font.hmtx_table == NULL);
        font.hmtx_table = GetTable(&font,"hmtx");
    }

    if (glyph_ids.size() == 0)
    {
        glyph_ids.clear();
        glyph_ids.reserve(font.numGlyphs);
        for (int x = 0; x < font.numGlyphs; ++x)
        {
            glyph_ids.push_back(x);
        }
    }
    else if (font.target_type == PS_TYPE_3 ||
             font.target_type == PS_TYPE_42_3_HYBRID)
    {
        ttfont_add_glyph_dependencies(&font, glyph_ids);
    }

} /* end of insert_ttfont() */

void insert_ttfont(const char *filename, TTStreamWriter& stream,
                   font_type_enum target_type, std::vector<int>& glyph_ids)
{
    struct TTFONT font;

    read_font(filename, target_type, glyph_ids, font);

    /* Write the header for the PostScript font. */
    ttfont_header(stream, &font);

    /* Define the encoding. */
    ttfont_encoding(stream, &font, glyph_ids, target_type);

    /* Insert FontInfo dictionary. */
    ttfont_FontInfo(stream, &font);

    /* If we are generating a type 42 font, */
    /* emmit the sfnts array. */
    if (font.target_type == PS_TYPE_42 ||
        font.target_type == PS_TYPE_42_3_HYBRID)
    {
        ttfont_sfnts(stream, &font);
    }

    /* Emmit the CharStrings array. */
    ttfont_CharStrings(stream, &font, glyph_ids);

    /* Send the font trailer. */
    ttfont_trailer(stream, &font);

} /* end of insert_ttfont() */

class StringStreamWriter : public TTStreamWriter
{
    std::ostringstream oss;

public:
    void write(const char* a)
    {
        oss << a;
    }

    std::string str()
    {
        return oss.str();
    }
};

void get_pdf_charprocs(const char *filename, std::vector<int>& glyph_ids, TTDictionaryCallback& dict)
{
    struct TTFONT font;

    read_font(filename, PDF_TYPE_3, glyph_ids, font);

    for (std::vector<int>::const_iterator i = glyph_ids.begin();
            i != glyph_ids.end(); ++i)
    {
        StringStreamWriter writer;
        tt_type3_charproc(writer, &font, *i);
        const char* name = ttfont_CharStrings_getname(&font, *i);
        dict.add_pair(name, writer.str().c_str());
    }
}

TTFONT::TTFONT() :
    file(NULL),
    PostName(NULL),
    FullName(NULL),
    FamilyName(NULL),
    Style(NULL),
    Copyright(NULL),
    Version(NULL),
    Trademark(NULL),
    offset_table(NULL),
    post_table(NULL),
    loca_table(NULL),
    glyf_table(NULL),
    hmtx_table(NULL)
{

}

TTFONT::~TTFONT()
{
    if (file)
    {
        fclose(file);
    }
    free(PostName);
    free(FullName);
    free(FamilyName);
    free(Style);
    free(Copyright);
    free(Version);
    free(Trademark);
    free(offset_table);
    free(post_table);
    free(loca_table);
    free(glyf_table);
    free(hmtx_table);
}

/* end of file */
