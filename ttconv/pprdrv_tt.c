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
#include <stdio.h>
#include <string.h>
#include "pprdrv.h"
#include "interface.h"
#include "truetype.h"

/*==========================================================================
** If we can find the TrueType font with the indicated PostScript name, 
** return the TrueType font file name, otherwise return NULL.
==========================================================================*/
void derr(void) { }

char *find_ttfont(char *name)
    {
    } /* end of find_ttfont() */

/*===============================================================================
** If this routine is called, we should make our best effort to provide
** a TrueType rasterizer.  If the PPD file contains a "*TTRasterizer: Type42"
** line then we need do nothing special.  If the PPD file contains a
** "*TTRasterizer: Accept68K" line then TrueType is ok only if the job contains
** TrueDict or we can arrange to have it downloaded.  If the PPD file contains
** any other "*TTRasterizer:" line or none at all then we must assume that
** TrueType fonts are not acceptable.
===============================================================================*/
void want_ttrasterizer(void)
    {
    } /* end of want_ttrasterizer() */

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

    for(x=0; x<4; x++)
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

    for(x=0; x<2; x++)
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
BYTE *GetTable(struct TTFONT *font, char *name)
    {
    BYTE *ptr;
    int x;
    
    #ifdef DEBUG_TRUETYPE
    debug("GetTable(file,font,\"%s\")",name);
    #endif

    /* We must search the table directory. */
    ptr = font->offset_table + 12;
    x=0;
    while(TRUE)
    	{
	if( strncmp(ptr,name,4) == 0 )
	    {
	    ULONG offset,length;
	    BYTE *table;

	    offset = getULONG( ptr + 8 );
	    length = getULONG( ptr + 12 );	    
	    table = myalloc( sizeof(BYTE), length );

	    #ifdef DEBUG_TRUETYPE
	    debug("Loading table \"%s\" from offset %d, %d bytes",name,offset,length);
	    #endif
	    
	    if( fseek( font->file, (long)offset, SEEK_SET ) )
	    	fatal(EXIT_TTFONT,"TrueType font may be corrupt (reason 3)");

	    if( fread(table,sizeof(BYTE),length,font->file) != (sizeof(BYTE) * length))
		fatal(EXIT_TTFONT,"TrueType font may be corrupt (reason 4)");
		
	    return table;
	    }

    	x++;
    	ptr += 16;
    	if(x == font->numTables)
	    fatal(EXIT_TTFONT,"TrueType font is missing table");
    	}

    } /* end of GetTable() */

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
    int numrecords;			/* Number of strings in this table */
    BYTE *strings;			/* pointer to start of string storage */
    int x;
    int platform,encoding;		/* Current platform id, encoding id, */
    int language,nameid;		/* language id, name id, */
    int offset,length;			/* offset and length of string. */
    
    #ifdef DEBUG_TRUETYPE
    debug("Read_name()");
    #endif

    /* Set default values to avoid future references to */
    /* undefined pointers. */
    font->PostName = font->FullName =
    	font->FamilyName = font->Version = font->Style = "unknown";
    font->Copyright = font->Trademark = (char*)NULL;

    table_ptr = GetTable(font,"name");		/* pointer to table */
    numrecords = getUSHORT( table_ptr + 2 );	/* number of names */
    strings = table_ptr + getUSHORT( table_ptr + 4 );	/* start of string storage */
    
    ptr2 = table_ptr + 6;
    for(x=0; x < numrecords; x++,ptr2+=12)
    	{
	platform = getUSHORT(ptr2);
	encoding = getUSHORT(ptr2+2);
	language = getUSHORT(ptr2+4);
	nameid = getUSHORT(ptr2+6);
	length = getUSHORT(ptr2+8);
	offset = getUSHORT(ptr2+10);

	#ifdef DEBUG_TRUETYPE
	debug("platform %d, encoding %d, language 0x%x, name %d, offset %d, length %d",
		platform,encoding,language,nameid,offset,length);
	#endif    	

	/* Copyright notice */
	if( platform == 1 && nameid == 0 )
	    {
	    font->Copyright = (char*)myalloc(sizeof(char),length+1);
	    strncpy(font->Copyright,strings+offset,length);
	    font->Copyright[length]=(char)NULL;
	    
	    #ifdef DEBUG_TRUETYPE
	    debug("font->Copyright=\"%s\"",font->Copyright);
	    #endif
	    continue;
	    }
	

	/* Font Family name */
	if( platform == 1 && nameid == 1 )
	    {
	    font->FamilyName = (char*)myalloc(sizeof(char),length+1);
	    strncpy(font->FamilyName,strings+offset,length);
	    font->FamilyName[length]=(char)NULL;
	    
	    #ifdef DEBUG_TRUETYPE
	    debug("font->FamilyName=\"%s\"",font->FamilyName);
	    #endif
	    continue;
	    }


	/* Font Family name */
	if( platform == 1 && nameid == 2 )
	    {
	    font->Style = (char*)myalloc(sizeof(char),length+1);
	    strncpy(font->Style,strings+offset,length);
	    font->Style[length]=(char)NULL;
	    
	    #ifdef DEBUG_TRUETYPE
	    debug("font->Style=\"%s\"",font->Style);
	    #endif
	    continue;
	    }


	/* Full Font name */
	if( platform == 1 && nameid == 4 )
	    {
	    font->FullName = (char*)myalloc(sizeof(char),length+1);
	    strncpy(font->FullName,strings+offset,length);
	    font->FullName[length]=(char)NULL;
	    
	    #ifdef DEBUG_TRUETYPE
	    debug("font->FullName=\"%s\"",font->FullName);
	    #endif
	    continue;
	    }


	/* Version string */
	if( platform == 1 && nameid == 5 )
	    {
	    font->Version = (char*)myalloc(sizeof(char),length+1);
	    strncpy(font->Version,strings+offset,length);
	    font->Version[length]=(char)NULL;
	    
	    #ifdef DEBUG_TRUETYPE
	    debug("font->Version=\"%s\"",font->Version);
	    #endif
	    continue;
	    }


	/* PostScript name */
	if( platform == 1 && nameid == 6 )
	    {
	    font->PostName = (char*)myalloc(sizeof(char),length+1);
	    strncpy(font->PostName,strings+offset,length);
	    font->PostName[length]=(char)NULL;
	    
	    #ifdef DEBUG_TRUETYPE
	    debug("font->PostName=\"%s\"",font->PostName);
	    #endif
	    continue;
	    }


	/* Trademark string */
	if( platform == 1 && nameid == 7 )
	    {
	    font->Trademark = (char*)myalloc(sizeof(char),length+1);
	    strncpy(font->Trademark,strings+offset,length);
	    font->Trademark[length]=(char)NULL;
	    
	    #ifdef DEBUG_TRUETYPE
	    debug("font->Trademark=\"%s\"",font->Trademark);
	    #endif
	    continue;
	    }

    	}

    myfree(table_ptr);
    } /* end of Read_name() */

/*---------------------------------------------------------------------
** Write the header for a PostScript font.
---------------------------------------------------------------------*/
void ttfont_header(struct TTFONT *font)
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
    if( font->target_type == 42 )
    	{
    	printf("%%!PS-TrueTypeFont-%d.%d-%d.%d\n",
    		font->TTVersion.whole, font->TTVersion.fraction,
    		font->MfrRevision.whole, font->MfrRevision.fraction);
    	}

    /* If it is not a Type 42 font, we will use a different format. */
    else
    	{
    	printer_putline("%!PS-Adobe-3.0 Resource-Font");
    	}	/* See RBIIp 641 */

    /* We will make the title the name of the font. */
    printf("%%%%Title: %s\n",font->FullName);

    /* If there is a Copyright notice, put it here too. */
    if( font->Copyright != (char*)NULL )
	printf("%%%%Copyright: %s\n",font->Copyright);

    /* We created this file. */
    if( font->target_type == 42 )
	printer_putline("%%Creator: Converted from TrueType to type 42 by PPR");
    else
	printer_putline("%%Creator: Converted from TrueType by PPR");

    /* If VM usage information is available, print it. */
    if( font->target_type == 42 )
    	{
	VMMin = (int)getULONG( font->post_table + 16 );
	VMMax = (int)getULONG( font->post_table + 20 );
	if( VMMin > 0 && VMMax > 0 )
	    printf("%%%%VMUsage: %d %d\n",VMMin,VMMax);
    	}

    /* Start the dictionary which will eventually */
    /* become the font. */
    if( font->target_type != 3 )
	{
	printer_putline("15 dict begin");
	}
    else
	{
	printer_putline("25 dict begin");

    	/* Type 3 fonts will need some subroutines here. */
	printer_putline("/_d{bind def}bind def");
	printer_putline("/_m{moveto}_d");
	printer_putline("/_l{lineto}_d");
	printer_putline("/_cl{closepath eofill}_d");
	printer_putline("/_c{curveto}_d");
	printer_putline("/_sc{7 -1 roll{setcachedevice}{pop pop pop pop pop pop}ifelse}_d");
	printer_putline("/_e{exec}_d");
	}

    printf("/FontName /%s def\n",font->PostName);
    printer_putline("/PaintType 0 def");

    if(font->target_type == 42)
	printer_putline("/FontMatrix[1 0 0 1 0 0]def");
    else
	printer_putline("/FontMatrix[.001 0 0 .001 0 0]def");

    printf("/FontBBox[%d %d %d %d]def\n",font->llx,font->lly,font->urx,font->ury);
    printf("/FontType %d def\n", font->target_type );
    } /* end of ttfont_header() */

/*-------------------------------------------------------------
** Define the encoding array for this font.
** It seems best to just use "Standard".
-------------------------------------------------------------*/
void ttfont_encoding(void)
    {
    printer_putline("/Encoding StandardEncoding def");
    } /* end of ttfont_encoding() */

/*-----------------------------------------------------------
** Create the optional "FontInfo" sub-dictionary.
-----------------------------------------------------------*/
void ttfont_FontInfo(struct TTFONT *font)
    {
    Fixed ItalicAngle;

    /* We create a sub dictionary named "FontInfo" where we */
    /* store information which though it is not used by the */
    /* interpreter, is useful to some programs which will */
    /* be printing with the font. */
    printer_putline("/FontInfo 10 dict dup begin");

    /* These names come from the TrueType font's "name" table. */
    printf("/FamilyName (%s) def\n",font->FamilyName);
    printf("/FullName (%s) def\n",font->FullName);

    if( font->Copyright != (char*)NULL || font->Trademark != (char*)NULL )
    	{
    	printf("/Notice (%s",
    		font->Copyright != (char*)NULL ? font->Copyright : "");
    	printf("%s%s) def\n",
    		font->Trademark != (char*)NULL ? " " : "",
    		font->Trademark != (char*)NULL ? font->Trademark : "");
    	}

    /* This information is not quite correct. */
    printf("/Weight (%s) def\n",font->Style);

    /* Some fonts have this as "version". */
    printf("/Version (%s) def\n",font->Version);

    /* Some information from the "post" table. */
    ItalicAngle = getFixed( font->post_table + 4 );
    printf("/ItalicAngle %d.%d def\n",ItalicAngle.whole,ItalicAngle.fraction);
    printf("/isFixedPitch %s def\n", getULONG( font->post_table + 12 ) ? "true" : "false" );    
    printf("/UnderlinePosition %d def\n", (int)getFWord( font->post_table + 8 ) );
    printf("/UnderlineThickness %d def\n", (int)getFWord( font->post_table + 10 ) );    
    printer_putline("end readonly def");    
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
void sfnts_start(void)
    {
    puts("/sfnts[<");
    in_string=TRUE;
    string_len=0;
    line_len=8;
    } /* end of sfnts_start() */

/*
** Write a BYTE as a hexadecimal value as part of the sfnts array.
*/
void sfnts_pputBYTE(BYTE n)
    {
    static const char hexdigits[]="0123456789ABCDEF";

    if(!in_string)
    	{
	printer_putc('<');
    	string_len=0;
    	line_len++;
    	in_string=TRUE;
    	}

    printer_putc( hexdigits[ n / 16 ] );
    printer_putc( hexdigits[ n % 16 ] );
    string_len++;
    line_len+=2;

    if(line_len > 70)
   	{
   	printer_putc('\n');
   	line_len=0;
   	}
   	
    } /* end of sfnts_pputBYTE() */
   
/*
** Write a USHORT as a hexadecimal value as part of the sfnts array.
*/
void sfnts_pputUSHORT(USHORT n)
    {
    sfnts_pputBYTE(n / 256);
    sfnts_pputBYTE(n % 256);
    } /* end of sfnts_pputUSHORT() */

/*
** Write a ULONG as part of the sfnts array.
*/
void sfnts_pputULONG(ULONG n)
    {
    int x1,x2,x3;
    
    x1 = n % 256;
    n /= 256;
    x2 = n % 256;
    n /= 256;
    x3 = n % 256;
    n /= 256;

    sfnts_pputBYTE(n);
    sfnts_pputBYTE(x3);
    sfnts_pputBYTE(x2);
    sfnts_pputBYTE(x1);
    } /* end of sfnts_pputULONG() */

/*
** This is called whenever it is 
** necessary to end a string in the sfnts array.
**
** (The array must be broken into strings which are
** no longer than 64K characters.)
*/
void sfnts_end_string(void)
    {
    if(in_string)
    	{
	string_len=0;		/* fool sfnts_pputBYTE() */
	
	#ifdef DEBUG_TRUETYPE_INLINE
	puts("\n% dummy byte:\n");
	#endif

	sfnts_pputBYTE(0);	/* extra byte for pre-2013 compatibility */
	printer_putc('>');
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
void sfnts_new_table(ULONG length)
    {
    if( (string_len + length) > 65528 )    
        sfnts_end_string();
    } /* end of sfnts_new_table() */

/*
** We may have to break up the 'glyf' table.  That is the reason
** why we provide this special routine to copy it into the sfnts
** array.
*/
void sfnts_glyf_table(struct TTFONT *font, ULONG oldoffset, ULONG correct_total_length)
    {
    int x;
    ULONG off;
    ULONG length;
    int c;
    ULONG total=0;		/* running total of bytes written to table */

    #ifdef DEBUG_TRUETYPE
    debug("sfnts_glyf_table(font,%d)", (int)correct_total_length);
    #endif

    font->loca_table = GetTable(font,"loca");

    /* Seek to proper position in the file. */
    fseek( font->file, oldoffset, SEEK_SET );

    /* Copy the glyphs one by one */
    for(x=0; x < font->numGlyphs; x++)
	{
	/* Read the glyph offset from the index-to-location table. */
	if(font->indexToLocFormat == 0)
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
	sfnts_new_table( (int)length );

	/* 
	** Make sure the glyph is padded out to a
	** two byte boundary.
	*/
	if( length % 2 )
	    fatal(EXIT_TTFONT,"TrueType font contains a 'glyf' table without 2 byte padding");

	/* Copy the bytes of the glyph. */
	while( length-- )
	    {
	    if( (c = fgetc(font->file)) == EOF )
	    	fatal(EXIT_TTFONT,"TrueType font may be corrupt (reason 6)");
	    
	    sfnts_pputBYTE(c);
	    total++;		/* add to running total */
	    }	    

	}

    myfree(font->loca_table);
    
    /* Pad out to full length from table directory */
    while( total < correct_total_length )
    	{
    	sfnts_pputBYTE(0);
    	total++;
    	}

    /* Look for unexplainable descrepancies between sizes */ 
    if( total != correct_total_length )
	{
    	fatal(EXIT_TTFONT,"pprdrv_tt.c: sfnts_glyf_table(): total != correct_total_length");
    	}

    } /* end of sfnts_glyf_table() */

/*
** Here is the routine which ties it all together.
**
** Create the array called "sfnts" which 
** holds the actual TrueType data.
*/
void ttfont_sfnts(struct TTFONT *font)
    {
    char *table_names[]=	/* The names of all tables */
    	{			/* which it is worth while */
    	"cvt ",			/* to include in a Type 42 */
    	"fpgm",			/* PostScript font. */
    	"glyf",
    	"head",
    	"hhea",
    	"hmtx",
    	"loca",
    	"maxp",
    	"prep"
    	} ;

    struct {			/* The location of each of */
    	ULONG oldoffset;	/* the above tables. */
    	ULONG newoffset;
    	ULONG length;
    	ULONG checksum;
    	} tables[9];
    	
    BYTE *ptr;			/* A pointer into the origional table directory. */
    int x,y;			/* General use loop countes. */
    int c;			/* Input character. */
    int diff;
    ULONG nextoffset;
    int count;			/* How many `important' tables did we find? */

    ptr = font->offset_table + 12;
    nextoffset=0;
    count=0;
    
    /* 
    ** Find the tables we want and store there vital
    ** statistics in tables[].
    */
    for(x=0; x < 9; x++ )
    	{
    	do  {
    	    diff = strncmp( (char*)ptr, table_names[x], 4 );

	    if( diff > 0 )		/* If we are past it. */
	    	{
		tables[x].length = 0;
		diff = 0;		
	    	}
	    else if( diff < 0 )		/* If we haven't hit it yet. */
	        {
	        ptr += 16;
	        }
	    else if( diff == 0 )	/* Here it is! */
	    	{
		tables[x].newoffset = nextoffset;
		tables[x].checksum = getULONG( ptr + 4 );
		tables[x].oldoffset = getULONG( ptr + 8 );
		tables[x].length = getULONG( ptr + 12 );
		nextoffset += ( ((tables[x].length + 3) / 4) * 4 );
		count++;
		ptr += 16;	    	
	    	}
    	    } while(diff != 0);
    	
    	} /* end of for loop which passes over the table directory */
    
    /* Begin the sfnts array. */
    sfnts_start();

    /* Generate the offset table header */
    /* Start by copying the TrueType version number. */
    ptr = font->offset_table;
    for(x=0; x < 4; x++)
	{ 
   	sfnts_pputBYTE( *(ptr++) );
   	}
    
    /* Now, generate those silly numTables numbers. */
    sfnts_pputUSHORT(count);		/* number of tables */
    if( count == 9 )
    	{
    	sfnts_pputUSHORT(7);		/* searchRange */
    	sfnts_pputUSHORT(3);		/* entrySelector */
    	sfnts_pputUSHORT(81);		/* rangeShift */
    	}    
    #ifdef DEBUG_TRUETYPE
    else
    	{
	debug("only %d tables selected",count); 	
    	}
    #endif

    /* Now, emmit the table directory. */
    for(x=0; x < 9; x++)
    	{
	if( tables[x].length == 0 )	/* Skip missing tables */
	    continue;

	/* Name */
	sfnts_pputBYTE( table_names[x][0] );
	sfnts_pputBYTE( table_names[x][1] );
	sfnts_pputBYTE( table_names[x][2] );
	sfnts_pputBYTE( table_names[x][3] );
	
	/* Checksum */
	sfnts_pputULONG( tables[x].checksum );

	/* Offset */
	sfnts_pputULONG( tables[x].newoffset + 12 + (count * 16) );
	
	/* Length */
	sfnts_pputULONG( tables[x].length );
    	}

    /* Now, send the tables */
    for(x=0; x < 9; x++)
    	{
    	if( tables[x].length == 0 )	/* skip tables that aren't there */
    	    continue;
    	    
	#ifdef DEBUG_TRUETYPE
	debug("emmiting table '%s'",table_names[x]);
	#endif

	/* 'glyf' table gets special treatment */
	if( strcmp(table_names[x],"glyf")==0 )
	    {
	    sfnts_glyf_table(font,tables[x].oldoffset,tables[x].length);
	    }
	else			/* Other tables may not exceed */
	    {			/* 65535 bytes in length. */
	    if( tables[x].length > 65535 )
	    	fatal(EXIT_TTFONT,"TrueType font has a table which is too long");	    
	    
	    /* Start new string if necessary. */
	    sfnts_new_table(tables[x].length);

	    /* Seek to proper position in the file. */
    	    fseek( font->file, tables[x].oldoffset, SEEK_SET );
    	
	    /* Copy the bytes of the table. */
	    for( y=0; y < tables[x].length; y++ )
	        {
	        if( (c = fgetc(font->file)) == EOF )
	    	    fatal(EXIT_TTFONT,"TrueType font may be corrupt (reason 7)");
	    
	        sfnts_pputBYTE(c);
	        }	    
	    }

	/* Padd it out to a four byte boundary. */
	y=tables[x].length;
	while( (y % 4) != 0 )
	    {
	    sfnts_pputBYTE(0);
	    y++;
	    #ifdef DEBUG_TRUETYPE_INLINE
	    puts("\n% pad byte:\n");
	    #endif
	    }

    	} /* End of loop for all tables */

    /* Close the array. */
    sfnts_end_string();    
    printer_putline("]def");
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
char *Apple_CharStrings[]={ 
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
"diamond","appleoutline"};

/*
** This routine is called by the one below.
** It is also called from pprdrv_tt2.c
*/
char *ttfont_CharStrings_getname(struct TTFONT *font, int charindex)
    {
    int GlyphIndex;
    static char temp[80];
    char *ptr;
    int len;

    GlyphIndex = (int)getUSHORT( font->post_table + 34 + (charindex * 2) );
	
    if( GlyphIndex <= 257 )		/* If a standard Apple name, */
	{
	return Apple_CharStrings[GlyphIndex];
	}
    else				/* Otherwise, use one */
	{				/* of the pascal strings. */
	GlyphIndex -= 258;
	    
	/* Set pointer to start of Pascal strings. */
	ptr = (char*)(font->post_table + 34 + (font->numGlyphs * 2));

	len = (int)*(ptr++);	/* Step thru the strings */
	while(GlyphIndex--)		/* until we get to the one */
	    {			/* that we want. */
	    ptr += len;
	    len = (int)*(ptr++);
	    }
		
	if( len >= sizeof(temp) )
	    fatal(EXIT_TTFONT,"TrueType font file contains a very long PostScript name");

	strncpy(temp,ptr,len);	/* Copy the pascal string into */
	temp[len]=(char)NULL;	/* a buffer and make it ASCIIz. */

	return temp;
	}
    } /* end of ttfont_CharStrings_getname() */

/*
** This is the central routine of this section.
*/
void ttfont_CharStrings(struct TTFONT *font)
    {
    Fixed post_format;
    int x;
    
    /* The 'post' table format number. */
    post_format = getFixed( font->post_table );
    
    if( post_format.whole != 2 || post_format.fraction != 0 )
    	fatal(EXIT_TTFONT,"TrueType fontdoes not have a format 2.0 'post' table");

    /* Emmit the start of the PostScript code to define the dictionary. */
    printf("/CharStrings %d dict dup begin\n", font->numGlyphs);
    
    /* Emmit one key-value pair for each glyph. */
    for(x=0; x < font->numGlyphs; x++)
    	{
	if(font->target_type == 42)	/* type 42 */
 	    {
 	    printf("/%s %d def\n",ttfont_CharStrings_getname(font,x),x);
	    }
	else				/* type 3 */
 	    {
 	    printf("/%s{",ttfont_CharStrings_getname(font,x));
	    	
	    tt_type3_charproc(font,x);
    	
	    printer_putline("}_d");	/* "} bind def" */
 	    }
    	}
    
    printer_putline("end readonly def");
    } /* end of ttfont_CharStrings() */

/*----------------------------------------------------------------
** Emmit the code to finish up the dictionary and turn
** it into a font.
----------------------------------------------------------------*/
void ttfont_trailer(struct TTFONT *font)
    {
    /* If we are generating a type 3 font, we need to provide */
    /* a BuildGlyph and BuildChar proceedures. */
    if( font->target_type == 3 )
    	{
	printer_putc('\n');

	printer_putline("/BuildGlyph");
	printer_putline(" {exch begin");		/* start font dictionary */
	printer_putline(" CharStrings exch");
	printer_putline(" 2 copy known not{pop /.notdef}if");
	printer_putline(" true 3 1 roll get exec");
	printer_putline(" end}_d");

	printer_putc('\n');

	/* This proceedure is for compatiblity with */
	/* level 1 interpreters. */
	printer_putline("/BuildChar {");
	printer_putline(" 1 index /Encoding get exch get");
	printer_putline(" 1 index /BuildGlyph get exec");
	printer_putline("}_d");    	
	
	printer_putc('\n');
    	}

    /* If we are generating a type 42 font, we need to check to see */
    /* if this PostScript interpreter understands type 42 fonts.  If */
    /* it doesn't, we will hope that the Apple TrueType rasterizer */
    /* has been loaded and we will adjust the font accordingly. */
    /* I found out how to do this by examining a TrueType font */
    /* generated by a Macintosh.  That is where the TrueType interpreter */
    /* setup instructions and part of BuildGlyph came from. */
    else if( font->target_type == 42 )
    	{
	printer_putc('\n');

	/* If we have no "resourcestatus" command, or FontType 42 */
	/* is unknown, leave "true" on the stack. */
	printer_putline("systemdict/resourcestatus known");
	printer_putline(" {42 /FontType resourcestatus");
	printer_putline("   {pop pop false}{true}ifelse}");
	printer_putline(" {true}ifelse");

	/* If true, execute code to produce an error message if */
	/* we can't find Apple's TrueDict in VM. */
	printer_putline("{/TrueDict where{pop}{(%%[ Error: no TrueType rasterizer ]%%)= flush}ifelse");

	/* Since we are expected to use Apple's TrueDict TrueType */
	/* reasterizer, change the font type to 3. */    	
    	printer_putline("/FontType 3 def");

	/* Define a string to hold the state of the Apple */
	/* TrueType interpreter. */
    	printer_putline(" /TrueState 271 string def");

	/* It looks like we get information about the resolution */
	/* of the printer and store it in the TrueState string. */
    	printer_putline(" TrueDict begin sfnts save");
    	printer_putline(" 72 0 matrix defaultmatrix dtransform dup");
    	printer_putline(" mul exch dup mul add sqrt cvi 0 72 matrix");
    	printer_putline(" defaultmatrix dtransform dup mul exch dup");
    	printer_putline(" mul add sqrt cvi 3 -1 roll restore");
    	printer_putline(" TrueState initer end");

	/* This BuildGlyph procedure will look the name up in the */
	/* CharStrings array, and then check to see if what it gets */
	/* is a procedure.  If it is, it executes it, otherwise, it */
	/* lets the TrueType rasterizer loose on it. */

	/* When this proceedure is executed the stack contains */
	/* the font dictionary and the character name.  We */
	/* exchange arguments and move the dictionary to the */
	/* dictionary stack. */
	printer_putline(" /BuildGlyph{exch begin");
		/* stack: charname */

	/* Put two copies of CharStrings on the stack and consume */
	/* one testing to see if the charname is defined in it, */
	/* leave the answer on the stack. */
	printer_putline("  CharStrings dup 2 index known");
		/* stack: charname CharStrings bool */

	/* Exchange the CharStrings dictionary and the charname, */
	/* but if the answer was false, replace the character name */
	/* with ".notdef". */ 
	printer_putline("    {exch}{exch pop /.notdef}ifelse");
		/* stack: CharStrings charname */

	/* Get the value from the CharStrings dictionary and see */
	/* if it is executable. */
	printer_putline("  get dup xcheck");
		/* stack: CharStrings_entry */

	/* If is a proceedure.  Execute according to RBIIp 277-278. */
	printer_putline("    {currentdict systemdict begin begin exec end end}");

	/* Is a TrueType character index, let the rasterizer at it. */
	printer_putline("    {TrueDict begin /bander load cvlit exch TrueState render end}");

	printer_putline("    ifelse");

	/* Pop the font's dictionary off the stack. */
	printer_putline(" end}bind def");

	/* This is the level 1 compatibility BuildChar procedure. */
	/* See RBIIp 281. */
	printer_putline(" /BuildChar{");
	printer_putline("  1 index /Encoding get exch get");
	printer_putline("  1 index /BuildGlyph get exec");
	printer_putline(" }bind def");    	

	/* Here we close the condition which is true */
	/* if the printer has no built-in TrueType */
	/* rasterizer. */
	printer_putline("}if");
	printer_putc('\n');
    	} /* end of if Type 42 not understood. */

    printer_putline("FontName currentdict end definefont pop");
    printer_putline("%%EOF");
    } /* end of ttfont_trailer() */    

/*------------------------------------------------------------------
** This is the externally callable routine which inserts the font.
------------------------------------------------------------------*/
void insert_ttfont(char *filename)
    {
    struct TTFONT font;
    BYTE *ptr;
    
    #ifdef DEBUG_TRUETYPE
    debug("insert_ttfont(\"%s\")",filename);
    #endif

    /* Decide what type of PostScript font we will be generating. */
    if( printer.type42_ok )
	font.target_type = 42;
    else
	font.target_type = 3;

    /* Save the file name for error messages. */
    font.filename=filename;

    /* Open the font file */
    if( (font.file = fopen(filename,"r")) == (FILE*)NULL )
    	fatal(EXIT_TTFONT,"Failed to open TrueType font");

    /* Allocate space for the unvarying part of the offset table. */
    font.offset_table = myalloc( 12, sizeof(BYTE) );
    
    /* Read the first part of the offset table. */
    if( fread( font.offset_table, sizeof(BYTE), 12, font.file ) != 12 )
    	fatal(EXIT_TTFONT,"TrueType font may be corrupt (reason 1)");
    
    /* Determine how many directory entries there are. */
    font.numTables = getUSHORT( font.offset_table + 4 );
    #ifdef DEBUG_TRUETYPE
    debug("numTables=%d",(int)font.numTables);
    #endif
    
    /* Expand the memory block to hold the whole thing. */
    font.offset_table = myrealloc( font.offset_table, sizeof(BYTE) * (12 + font.numTables * 16) );
    
    /* Read the rest of the table directory. */
    if( fread( font.offset_table + 12, sizeof(BYTE), (font.numTables*16), font.file ) != (font.numTables*16) )
    	fatal(EXIT_TTFONT,"TrueType font may be corrupt (reason 2)");
    
    /* Extract information from the "Offset" table. */
    font.TTVersion = getFixed( font.offset_table );

    /* Load the "head" table and extract information from it. */
    ptr = GetTable(&font,"head");
    font.MfrRevision = getFixed( ptr + 4 );		/* font revision number */
    font.unitsPerEm = getUSHORT( ptr + 18 );
    font.HUPM = font.unitsPerEm / 2;
    #ifdef DEBUG_TRUETYPE
    debug("unitsPerEm=%d",(int)font.unitsPerEm);
    #endif
    font.llx = topost2( getFWord( ptr + 36 ) );		/* bounding box info */
    font.lly = topost2( getFWord( ptr + 38 ) );
    font.urx = topost2( getFWord( ptr + 40 ) );
    font.ury = topost2( getFWord( ptr + 42 ) );
    font.indexToLocFormat = getSHORT( ptr + 50 );	/* size of 'loca' data */
    if(font.indexToLocFormat != 0 && font.indexToLocFormat != 1)
    	fatal(EXIT_TTFONT,"TrueType font is unusable because indexToLocFormat != 0");
    if( getSHORT(ptr+52) != 0 )
    	fatal(EXIT_TTFONT,"TrueType font is unusable because glyphDataFormat != 0");
    myfree(ptr);

    /* Load information from the "name" table. */
    Read_name(&font);

    /* We need to have the PostScript table around. */
    font.post_table = GetTable(&font,"post");
    font.numGlyphs = getUSHORT( font.post_table + 32 );

    /* Write the header for the PostScript font. */
    ttfont_header(&font);

    /* Define the encoding. */
    ttfont_encoding();

    /* Insert FontInfo dictionary. */
    ttfont_FontInfo(&font);

    /* If we are generating a type 42 font, */
    /* emmit the sfnts array. */
    if( font.target_type == 42 )
	ttfont_sfnts(&font);

    /* If we are generating a Type 3 font, we will need to */
    /* have the 'loca' and 'glyf' tables arround while */
    /* we are generating the CharStrings. */
    if(font.target_type == 3)
    	{
	BYTE *ptr;			/* We need only one value */
	ptr = GetTable(&font,"hhea");
	font.numberOfHMetrics = getUSHORT(ptr + 34);
	myfree(ptr);

	font.loca_table = GetTable(&font,"loca");
	font.glyf_table = GetTable(&font,"glyf");
	font.hmtx_table = GetTable(&font,"hmtx");
	}

    /* Emmit the CharStrings array. */
    ttfont_CharStrings(&font);

    /* Free the space occupied by the 'loca' and 'glyf' tables */
    /* if we loaded them. */
    if(font.target_type == 3)
    	{
    	myfree(font.loca_table);
    	myfree(font.glyf_table);
	myfree(font.hmtx_table);
    	}

    /* Send the font trailer. */
    ttfont_trailer(&font);

    /* We are done with the TrueType font file. */
    fclose(font.file);

    /* Free the memory occupied by tables. */
    myfree(font.offset_table);
    myfree(font.post_table);
    } /* end of insert_ttfont() */

/* end of file */
