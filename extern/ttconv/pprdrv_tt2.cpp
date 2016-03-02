/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */

/*
** ~ppr/src/pprdrv/pprdrv_tt2.c
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
** The functions in this file do most of the work to convert a
** TrueType font to a type 3 PostScript font.
**
** Most of the material in this file is derived from a program called
** "ttf2ps" which L. S. Ng posted to the usenet news group
** "comp.sources.postscript".  The author did not provide a copyright
** notice or indicate any restrictions on use.
**
** Last revised 11 July 1995.
*/

#include "global_defines.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <memory>
#include "pprdrv.h"
#include "truetype.h"
#include <algorithm>
#include <stack>
#include <list>

class GlyphToType3
{
private:
    GlyphToType3& operator=(const GlyphToType3& other);
    GlyphToType3(const GlyphToType3& other);

    /* The PostScript bounding box. */
    int llx,lly,urx,ury;
    int advance_width;

    /* Variables to hold the character data. */
    int *epts_ctr;                      /* array of contour endpoints */
    int num_pts, num_ctr;               /* number of points, number of coutours */
    FWord *xcoor, *ycoor;               /* arrays of x and y coordinates */
    BYTE *tt_flags;                     /* array of TrueType flags */

    int stack_depth;            /* A book-keeping variable for keeping track of the depth of the PS stack */

    bool pdf_mode;

    void load_char(TTFONT* font, BYTE *glyph);
    void stack(TTStreamWriter& stream, int new_elem);
    void stack_end(TTStreamWriter& stream);
    void PSConvert(TTStreamWriter& stream);
    void PSCurveto(TTStreamWriter& stream,
                   FWord x0, FWord y0,
                   FWord x1, FWord y1,
                   FWord x2, FWord y2);
    void PSMoveto(TTStreamWriter& stream, int x, int y);
    void PSLineto(TTStreamWriter& stream, int x, int y);
    void do_composite(TTStreamWriter& stream, struct TTFONT *font, BYTE *glyph);

public:
    GlyphToType3(TTStreamWriter& stream, struct TTFONT *font, int charindex, bool embedded = false);
    ~GlyphToType3();
};

// Each point on a TrueType contour is either on the path or off it (a
// control point); here's a simple representation for building such
// contours. Added by Jouni Seppänen 2012-05-27.
enum Flag { ON_PATH, OFF_PATH };
struct FlaggedPoint
{
    enum Flag flag;
    FWord x;
    FWord y;
    FlaggedPoint(Flag flag_, FWord x_, FWord y_): flag(flag_), x(x_), y(y_) {};
};

double area(FWord *x, FWord *y, int n);
#define sqr(x) ((x)*(x))

#define NOMOREINCTR -1
#define NOMOREOUTCTR -1

/*
** This routine is used to break the character
** procedure up into a number of smaller
** procedures.  This is necessary so as not to
** overflow the stack on certain level 1 interpreters.
**
** Prepare to push another item onto the stack,
** starting a new proceedure if necessary.
**
** Not all the stack depth calculations in this routine
** are perfectly accurate, but they do the job.
*/
void GlyphToType3::stack(TTStreamWriter& stream, int new_elem)
{
    if ( !pdf_mode && num_pts > 25 )                    /* Only do something of we will */
    {
        /* have a log of points. */
        if (stack_depth == 0)
        {
            stream.put_char('{');
            stack_depth=1;
        }

        stack_depth += new_elem;                /* Account for what we propose to add */

        if (stack_depth > 100)
        {
            stream.puts("}_e{");
            stack_depth = 3 + new_elem; /* A rough estimate */
        }
    }
} /* end of stack() */

void GlyphToType3::stack_end(TTStreamWriter& stream)                    /* called at end */
{
    if ( !pdf_mode && stack_depth )
    {
        stream.puts("}_e");
        stack_depth=0;
    }
} /* end of stack_end() */

/*
** We call this routine to emmit the PostScript code
** for the character we have loaded with load_char().
*/
void GlyphToType3::PSConvert(TTStreamWriter& stream)
{
    int j, k;

    /* Step thru the contours.
     * j = index to xcoor, ycoor, tt_flags (point data)
     * k = index to epts_ctr (which points belong to the same contour) */
    for(j = k = 0; k < num_ctr; k++)
    {
        // A TrueType contour consists of on-path and off-path points.
        // Two consecutive on-path points are to be joined with a
        // line; off-path points between on-path points indicate a
        // quadratic spline, where the off-path point is the control
        // point. Two consecutive off-path points have an implicit
        // on-path point midway between them.
        std::list<FlaggedPoint> points;

        // Represent flags and x/y coordinates as a C++ list
        for (; j <= epts_ctr[k]; j++)
        {
            if (!(tt_flags[j] & 1)) {
                points.push_back(FlaggedPoint(OFF_PATH, xcoor[j], ycoor[j]));
            } else {
                points.push_back(FlaggedPoint(ON_PATH, xcoor[j], ycoor[j]));
            }
        }

        if (points.size() == 0) {
            // Don't try to access the last element of an empty list
            continue;
        }

        // For any two consecutive off-path points, insert the implied
        // on-path point.
        FlaggedPoint prev = points.back();
        for (std::list<FlaggedPoint>::iterator it = points.begin();
             it != points.end();
             it++)
        {
            if (prev.flag == OFF_PATH && it->flag == OFF_PATH)
            {
                points.insert(it,
                              FlaggedPoint(ON_PATH,
                                           (prev.x + it->x) / 2,
                                           (prev.y + it->y) / 2));
            }
            prev = *it;
        }
        // Handle the wrap-around: insert a point either at the beginning
        // or at the end that has the same coordinates as the opposite point.
        // This also ensures that the initial point is ON_PATH.
        if (points.front().flag == OFF_PATH)
        {
            assert(points.back().flag == ON_PATH);
            points.insert(points.begin(), points.back());
        }
        else
        {
            assert(points.front().flag == ON_PATH);
            points.push_back(points.front());
        }

        // The first point
        stack(stream, 3);
        PSMoveto(stream, points.front().x, points.front().y);

        // Step through the remaining points
        std::list<FlaggedPoint>::const_iterator it = points.begin();
        for (it++; it != points.end(); /* incremented inside */)
        {
            const FlaggedPoint& point = *it;
            if (point.flag == ON_PATH)
            {
                stack(stream, 3);
                PSLineto(stream, point.x, point.y);
                it++;
            } else {
                std::list<FlaggedPoint>::const_iterator prev = it, next = it;
                prev--;
                next++;
                assert(prev->flag == ON_PATH);
                assert(next->flag == ON_PATH);
                stack(stream, 7);
                PSCurveto(stream,
                          prev->x, prev->y,
                          point.x, point.y,
                          next->x, next->y);
                it++;
                it++;
            }
        }
    }

    /* Now, we can fill the whole thing. */
    stack(stream, 1);
    stream.puts( pdf_mode ? "f" : "_cl" );
} /* end of PSConvert() */

void GlyphToType3::PSMoveto(TTStreamWriter& stream, int x, int y)
{
    stream.printf(pdf_mode ? "%d %d m\n" : "%d %d _m\n",
                  x, y);
}

void GlyphToType3::PSLineto(TTStreamWriter& stream, int x, int y)
{
    stream.printf(pdf_mode ? "%d %d l\n" : "%d %d _l\n",
                  x, y);
}

/*
** Emit a PostScript "curveto" command, assuming the current point
** is (x0, y0), the control point of a quadratic spline is (x1, y1),
** and the endpoint is (x2, y2). Note that this requires a conversion,
** since PostScript splines are cubic.
*/
void GlyphToType3::PSCurveto(TTStreamWriter& stream,
                             FWord x0, FWord y0,
                             FWord x1, FWord y1,
                             FWord x2, FWord y2)
{
    double sx[3], sy[3], cx[3], cy[3];

    sx[0] = x0;
    sy[0] = y0;
    sx[1] = x1;
    sy[1] = y1;
    sx[2] = x2;
    sy[2] = y2;
    cx[0] = (2*sx[1]+sx[0])/3;
    cy[0] = (2*sy[1]+sy[0])/3;
    cx[1] = (sx[2]+2*sx[1])/3;
    cy[1] = (sy[2]+2*sy[1])/3;
    cx[2] = sx[2];
    cy[2] = sy[2];
    stream.printf("%d %d %d %d %d %d %s\n",
                  (int)cx[0], (int)cy[0], (int)cx[1], (int)cy[1],
                  (int)cx[2], (int)cy[2], pdf_mode ? "c" : "_c");
}

/*
** Deallocate the structures which stored
** the data for the last simple glyph.
*/
GlyphToType3::~GlyphToType3()
{
    free(tt_flags);            /* The flags array */
    free(xcoor);               /* The X coordinates */
    free(ycoor);               /* The Y coordinates */
    free(epts_ctr);            /* The array of contour endpoints */
}

/*
** Load the simple glyph data pointed to by glyph.
** The pointer "glyph" should point 10 bytes into
** the glyph data.
*/
void GlyphToType3::load_char(TTFONT* font, BYTE *glyph)
{
    int x;
    BYTE c, ct;

    /* Read the contour endpoints list. */
    epts_ctr = (int *)calloc(num_ctr,sizeof(int));
    for (x = 0; x < num_ctr; x++)
    {
        epts_ctr[x] = getUSHORT(glyph);
        glyph += 2;
    }

    /* From the endpoint of the last contour, we can */
    /* determine the number of points. */
    num_pts = epts_ctr[num_ctr-1]+1;
#ifdef DEBUG_TRUETYPE
    debug("num_pts=%d",num_pts);
    stream.printf("%% num_pts=%d\n",num_pts);
#endif

    /* Skip the instructions. */
    x = getUSHORT(glyph);
    glyph += 2;
    glyph += x;

    /* Allocate space to hold the data. */
    tt_flags = (BYTE *)calloc(num_pts,sizeof(BYTE));
    xcoor = (FWord *)calloc(num_pts,sizeof(FWord));
    ycoor = (FWord *)calloc(num_pts,sizeof(FWord));

    /* Read the flags array, uncompressing it as we go. */
    /* There is danger of overflow here. */
    for (x = 0; x < num_pts; )
    {
        tt_flags[x++] = c = *(glyph++);

        if (c&8)                /* If next byte is repeat count, */
        {
            ct = *(glyph++);

            if ( (x + ct) > num_pts )
            {
                throw TTException("Error in TT flags");
            }

            while (ct--)
            {
                tt_flags[x++] = c;
            }
        }
    }

    /* Read the x coordinates */
    for (x = 0; x < num_pts; x++)
    {
        if (tt_flags[x] & 2)            /* one byte value with */
        {
            /* external sign */
            c = *(glyph++);
            xcoor[x] = (tt_flags[x] & 0x10) ? c : (-1 * (int)c);
        }
        else if (tt_flags[x] & 0x10)    /* repeat last */
        {
            xcoor[x] = 0;
        }
        else                            /* two byte signed value */
        {
            xcoor[x] = getFWord(glyph);
            glyph+=2;
        }
    }

    /* Read the y coordinates */
    for (x = 0; x < num_pts; x++)
    {
        if (tt_flags[x] & 4)            /* one byte value with */
        {
            /* external sign */
            c = *(glyph++);
            ycoor[x] = (tt_flags[x] & 0x20) ? c : (-1 * (int)c);
        }
        else if (tt_flags[x] & 0x20)    /* repeat last value */
        {
            ycoor[x] = 0;
        }
        else                            /* two byte signed value */
        {
            ycoor[x] = getUSHORT(glyph);
            glyph+=2;
        }
    }

    /* Convert delta values to absolute values. */
    for (x = 1; x < num_pts; x++)
    {
        xcoor[x] += xcoor[x-1];
        ycoor[x] += ycoor[x-1];
    }

    for (x=0; x < num_pts; x++)
    {
        xcoor[x] = topost(xcoor[x]);
        ycoor[x] = topost(ycoor[x]);
    }

} /* end of load_char() */

/*
** Emmit PostScript code for a composite character.
*/
void GlyphToType3::do_composite(TTStreamWriter& stream, struct TTFONT *font, BYTE *glyph)
{
    USHORT flags;
    USHORT glyphIndex;
    int arg1;
    int arg2;

    /* Once around this loop for each component. */
    do
    {
        flags = getUSHORT(glyph);       /* read the flags word */
        glyph += 2;

        glyphIndex = getUSHORT(glyph);  /* read the glyphindex word */
        glyph += 2;

        if (flags & ARG_1_AND_2_ARE_WORDS)
        {
            /* The tt spec. seems to say these are signed. */
            arg1 = getSHORT(glyph);
            glyph += 2;
            arg2 = getSHORT(glyph);
            glyph += 2;
        }
        else                    /* The tt spec. does not clearly indicate */
        {
            /* whether these values are signed or not. */
            arg1 = *(signed char *)(glyph++);
            arg2 = *(signed char *)(glyph++);
        }

        if (flags & WE_HAVE_A_SCALE)
        {
            glyph += 2;
        }
        else if (flags & WE_HAVE_AN_X_AND_Y_SCALE)
        {
            glyph += 4;
        }
        else if (flags & WE_HAVE_A_TWO_BY_TWO)
        {
            glyph += 8;
        }
        else
        {
        }

        /* Debugging */
#ifdef DEBUG_TRUETYPE
        stream.printf("%% flags=%d, arg1=%d, arg2=%d\n",
                      (int)flags,arg1,arg2);
#endif

        if (pdf_mode)
        {
            if ( flags & ARGS_ARE_XY_VALUES )
            {
                /* We should have been able to use 'Do' to reference the
                   subglyph here.  However, that doesn't seem to work with
                   xpdf or gs (only acrobat), so instead, this just includes
                   the subglyph here inline. */
                stream.printf("q 1 0 0 1 %d %d cm\n", topost(arg1), topost(arg2));
            }
            else
            {
                stream.printf("%% unimplemented shift, arg1=%d, arg2=%d\n",arg1,arg2);
            }
            GlyphToType3(stream, font, glyphIndex, true);
            if ( flags & ARGS_ARE_XY_VALUES )
            {
                stream.printf("\nQ\n");
            }
        }
        else
        {
            /* If we have an (X,Y) shif and it is non-zero, */
            /* translate the coordinate system. */
            if ( flags & ARGS_ARE_XY_VALUES )
            {
                if ( arg1 != 0 || arg2 != 0 )
                    stream.printf("gsave %d %d translate\n", topost(arg1), topost(arg2) );
            }
            else
            {
                stream.printf("%% unimplemented shift, arg1=%d, arg2=%d\n",arg1,arg2);
            }

            /* Invoke the CharStrings procedure to print the component. */
            stream.printf("false CharStrings /%s get exec\n",
                          ttfont_CharStrings_getname(font,glyphIndex));

            /* If we translated the coordinate system, */
            /* put it back the way it was. */
            if ( flags & ARGS_ARE_XY_VALUES && (arg1 != 0 || arg2 != 0) )
            {
                stream.puts("grestore ");
            }
        }

    }
    while (flags & MORE_COMPONENTS);

} /* end of do_composite() */

/*
** Return a pointer to a specific glyph's data.
*/
BYTE *find_glyph_data(struct TTFONT *font, int charindex)
{
    ULONG off;
    ULONG length;

    /* Read the glyph offset from the index to location table. */
    if (font->indexToLocFormat == 0)
    {
        off = getUSHORT( font->loca_table + (charindex * 2) );
        off *= 2;
        length = getUSHORT( font->loca_table + ((charindex+1) * 2) );
        length *= 2;
        length -= off;
    }
    else
    {
        off = getULONG( font->loca_table + (charindex * 4) );
        length = getULONG( font->loca_table + ((charindex+1) * 4) );
        length -= off;
    }

    if (length > 0)
    {
        return font->glyf_table + off;
    }
    else
    {
        return (BYTE*)NULL;
    }

} /* end of find_glyph_data() */

GlyphToType3::GlyphToType3(TTStreamWriter& stream, struct TTFONT *font, int charindex, bool embedded /* = false */)
{
    BYTE *glyph;

    tt_flags = NULL;
    xcoor = NULL;
    ycoor = NULL;
    epts_ctr = NULL;
    stack_depth = 0;
    pdf_mode = font->target_type < 0;

    /* Get a pointer to the data. */
    glyph = find_glyph_data( font, charindex );

    /* If the character is blank, it has no bounding box, */
    /* otherwise read the bounding box. */
    if ( glyph == (BYTE*)NULL )
    {
        llx=lly=urx=ury=0;      /* A blank char has an all zero BoundingBox */
        num_ctr=0;              /* Set this for later if()s */
    }
    else
    {
        /* Read the number of contours. */
        num_ctr = getSHORT(glyph);

        /* Read PostScript bounding box. */
        llx = getFWord(glyph + 2);
        lly = getFWord(glyph + 4);
        urx = getFWord(glyph + 6);
        ury = getFWord(glyph + 8);

        /* Advance the pointer. */
        glyph += 10;
    }

    /* If it is a simple character, load its data. */
    if (num_ctr > 0)
    {
        load_char(font, glyph);
    }
    else
    {
        num_pts=0;
    }

    /* Consult the horizontal metrics table to determine */
    /* the character width. */
    if ( charindex < font->numberOfHMetrics )
    {
        advance_width = getuFWord( font->hmtx_table + (charindex * 4) );
    }
    else
    {
        advance_width = getuFWord( font->hmtx_table + ((font->numberOfHMetrics-1) * 4) );
    }

    /* Execute setcachedevice in order to inform the font machinery */
    /* of the character bounding box and advance width. */
    stack(stream, 7);
    if (pdf_mode)
    {
        if (!embedded) {
            stream.printf("%d 0 %d %d %d %d d1\n",
                          topost(advance_width),
                          topost(llx), topost(lly), topost(urx), topost(ury) );
        }
    }
    else if (font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("pop gsave .001 .001 scale %d 0 %d %d %d %d setcachedevice\n",
                      topost(advance_width),
                      topost(llx), topost(lly), topost(urx), topost(ury) );
    }
    else
    {
        stream.printf("%d 0 %d %d %d %d _sc\n",
                      topost(advance_width),
                      topost(llx), topost(lly), topost(urx), topost(ury) );
    }

    /* If it is a simple glyph, convert it, */
    /* otherwise, close the stack business. */
    if ( num_ctr > 0 )          /* simple */
    {
        PSConvert(stream);
    }
    else if ( num_ctr < 0 )     /* composite */
    {
        do_composite(stream, font, glyph);
    }

    if (font->target_type == PS_TYPE_42_3_HYBRID)
    {
        stream.printf("\ngrestore\n");
    }

    stack_end(stream);
}

/*
** This is the routine which is called from pprdrv_tt.c.
*/
void tt_type3_charproc(TTStreamWriter& stream, struct TTFONT *font, int charindex)
{
    GlyphToType3 glyph(stream, font, charindex);
} /* end of tt_type3_charproc() */

/*
** Some of the given glyph ids may refer to composite glyphs.
** This function adds all of the dependencies of those composite
** glyphs to the glyph id vector.  Michael Droettboom [06-07-07]
*/
void ttfont_add_glyph_dependencies(struct TTFONT *font, std::vector<int>& glyph_ids)
{
    std::sort(glyph_ids.begin(), glyph_ids.end());

    std::stack<int> glyph_stack;
    for (std::vector<int>::iterator i = glyph_ids.begin();
            i != glyph_ids.end(); ++i)
    {
        glyph_stack.push(*i);
    }

    while (glyph_stack.size())
    {
        int gind = glyph_stack.top();
        glyph_stack.pop();

        BYTE* glyph = find_glyph_data( font, gind );
        if (glyph != (BYTE*)NULL)
        {

            int num_ctr = getSHORT(glyph);
            if (num_ctr <= 0)   // This is a composite glyph
            {

                glyph += 10;
                USHORT flags = 0;

                do
                {
                    flags = getUSHORT(glyph);
                    glyph += 2;
                    gind = (int)getUSHORT(glyph);
                    glyph += 2;

                    std::vector<int>::iterator insertion =
                        std::lower_bound(glyph_ids.begin(), glyph_ids.end(), gind);
                    if (insertion == glyph_ids.end() || *insertion != gind)
                    {
                        glyph_ids.insert(insertion, gind);
                        glyph_stack.push(gind);
                    }

                    if (flags & ARG_1_AND_2_ARE_WORDS)
                    {
                        glyph += 4;
                    }
                    else
                    {
                        glyph += 2;
                    }

                    if (flags & WE_HAVE_A_SCALE)
                    {
                        glyph += 2;
                    }
                    else if (flags & WE_HAVE_AN_X_AND_Y_SCALE)
                    {
                        glyph += 4;
                    }
                    else if (flags & WE_HAVE_A_TWO_BY_TWO)
                    {
                        glyph += 8;
                    }
                }
                while (flags & MORE_COMPONENTS);
            }
        }
    }
}

/* end of file */
