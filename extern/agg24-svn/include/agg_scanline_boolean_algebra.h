//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------

#ifndef AGG_SCANLINE_BOOLEAN_ALGEBRA_INCLUDED
#define AGG_SCANLINE_BOOLEAN_ALGEBRA_INCLUDED

#include <stdlib.h>
#include <math.h>
#include "agg_basics.h"


namespace agg
{

    //-----------------------------------------------sbool_combine_spans_bin
    // Functor.
    // Combine two binary encoded spans, i.e., when we don't have any
    // anti-aliasing information, but only X and Length. The function
    // is compatible with any type of scanlines.
    //----------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline> 
    struct sbool_combine_spans_bin
    {
        void operator () (const typename Scanline1::const_iterator&, 
                          const typename Scanline2::const_iterator&, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            sl.add_span(x, len, cover_full);
        }
    };



    //---------------------------------------------sbool_combine_spans_empty
    // Functor.
    // Combine two spans as empty ones. The functor does nothing
    // and is used to XOR binary spans.
    //----------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline> 
    struct sbool_combine_spans_empty
    {
        void operator () (const typename Scanline1::const_iterator&, 
                          const typename Scanline2::const_iterator&, 
                          int, unsigned, 
                          Scanline&) const
        {}
    };



    //--------------------------------------------------sbool_add_span_empty
    // Functor.
    // Add nothing. Used in conbine_shapes_sub
    //----------------
    template<class Scanline1, 
             class Scanline> 
    struct sbool_add_span_empty
    {
        void operator () (const typename Scanline1::const_iterator&, 
                          int, unsigned, 
                          Scanline&) const
        {}
    };


    //----------------------------------------------------sbool_add_span_bin
    // Functor.
    // Add a binary span
    //----------------
    template<class Scanline1, 
             class Scanline> 
    struct sbool_add_span_bin
    {
        void operator () (const typename Scanline1::const_iterator&, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            sl.add_span(x, len, cover_full);
        }
    };

    


    //-----------------------------------------------------sbool_add_span_aa
    // Functor.
    // Add an anti-aliased span
    // anti-aliasing information, but only X and Length. The function
    // is compatible with any type of scanlines.
    //----------------
    template<class Scanline1, 
             class Scanline> 
    struct sbool_add_span_aa
    {
        void operator () (const typename Scanline1::const_iterator& span, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            if(span->len < 0)
            {
                sl.add_span(x, len, *span->covers);
            }
            else
            if(span->len > 0)
            {
                const typename Scanline1::cover_type* covers = span->covers;
                if(span->x < x) covers += x - span->x;
                sl.add_cells(x, len, covers);
            }
        }
    };




    //----------------------------------------------sbool_intersect_spans_aa
    // Functor.
    // Intersect two spans preserving the anti-aliasing information.
    // The result is added to the "sl" scanline.
    //------------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             unsigned CoverShift = cover_shift> 
    struct sbool_intersect_spans_aa
    {
        enum cover_scale_e
        {
            cover_shift = CoverShift,
            cover_size  = 1 << cover_shift,
            cover_mask  = cover_size - 1,
            cover_full  = cover_mask
        };
        

        void operator () (const typename Scanline1::const_iterator& span1, 
                          const typename Scanline2::const_iterator& span2, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            unsigned cover;
            const typename Scanline1::cover_type* covers1;
            const typename Scanline2::cover_type* covers2;

            // Calculate the operation code and choose the 
            // proper combination algorithm.
            // 0 = Both spans are of AA type
            // 1 = span1 is solid, span2 is AA
            // 2 = span1 is AA, span2 is solid
            // 3 = Both spans are of solid type
            //-----------------
            switch((span1->len < 0) | ((span2->len < 0) << 1))
            {
            case 0:      // Both are AA spans
                covers1 = span1->covers;
                covers2 = span2->covers;
                if(span1->x < x) covers1 += x - span1->x;
                if(span2->x < x) covers2 += x - span2->x;
                do
                {
                    cover = *covers1++ * *covers2++;
                    sl.add_cell(x++, 
                                (cover == cover_full * cover_full) ?
                                cover_full : 
                                (cover >> cover_shift));
                }
                while(--len);
                break;

            case 1:      // span1 is solid, span2 is AA
                covers2 = span2->covers;
                if(span2->x < x) covers2 += x - span2->x;
                if(*(span1->covers) == cover_full)
                {
                    sl.add_cells(x, len, covers2);
                }
                else
                {
                    do
                    {
                        cover = *(span1->covers) * *covers2++;
                        sl.add_cell(x++, 
                                    (cover == cover_full * cover_full) ?
                                    cover_full : 
                                    (cover >> cover_shift));
                    }
                    while(--len);
                }
                break;

            case 2:      // span1 is AA, span2 is solid
                covers1 = span1->covers;
                if(span1->x < x) covers1 += x - span1->x;
                if(*(span2->covers) == cover_full)
                {
                    sl.add_cells(x, len, covers1);
                }
                else
                {
                    do
                    {
                        cover = *covers1++ * *(span2->covers);
                        sl.add_cell(x++, 
                                    (cover == cover_full * cover_full) ?
                                    cover_full : 
                                    (cover >> cover_shift));
                    }
                    while(--len);
                }
                break;

            case 3:      // Both are solid spans
                cover = *(span1->covers) * *(span2->covers);
                sl.add_span(x, len, 
                            (cover == cover_full * cover_full) ?
                            cover_full : 
                            (cover >> cover_shift));
                break;
            }
        }
    };






    //--------------------------------------------------sbool_unite_spans_aa
    // Functor.
    // Unite two spans preserving the anti-aliasing information.
    // The result is added to the "sl" scanline.
    //------------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             unsigned CoverShift = cover_shift> 
    struct sbool_unite_spans_aa
    {
        enum cover_scale_e
        {
            cover_shift = CoverShift,
            cover_size  = 1 << cover_shift,
            cover_mask  = cover_size - 1,
            cover_full  = cover_mask
        };
        

        void operator () (const typename Scanline1::const_iterator& span1, 
                          const typename Scanline2::const_iterator& span2, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            unsigned cover;
            const typename Scanline1::cover_type* covers1;
            const typename Scanline2::cover_type* covers2;

            // Calculate the operation code and choose the 
            // proper combination algorithm.
            // 0 = Both spans are of AA type
            // 1 = span1 is solid, span2 is AA
            // 2 = span1 is AA, span2 is solid
            // 3 = Both spans are of solid type
            //-----------------
            switch((span1->len < 0) | ((span2->len < 0) << 1))
            {
            case 0:      // Both are AA spans
                covers1 = span1->covers;
                covers2 = span2->covers;
                if(span1->x < x) covers1 += x - span1->x;
                if(span2->x < x) covers2 += x - span2->x;
                do
                {
                    cover = cover_mask * cover_mask - 
                                (cover_mask - *covers1++) * 
                                (cover_mask - *covers2++);
                    sl.add_cell(x++, 
                                (cover == cover_full * cover_full) ?
                                cover_full : 
                                (cover >> cover_shift));
                }
                while(--len);
                break;

            case 1:      // span1 is solid, span2 is AA
                covers2 = span2->covers;
                if(span2->x < x) covers2 += x - span2->x;
                if(*(span1->covers) == cover_full)
                {
                    sl.add_span(x, len, cover_full);
                }
                else
                {
                    do
                    {
                        cover = cover_mask * cover_mask - 
                                    (cover_mask - *(span1->covers)) * 
                                    (cover_mask - *covers2++);
                        sl.add_cell(x++, 
                                    (cover == cover_full * cover_full) ?
                                    cover_full : 
                                    (cover >> cover_shift));
                    }
                    while(--len);
                }
                break;

            case 2:      // span1 is AA, span2 is solid
                covers1 = span1->covers;
                if(span1->x < x) covers1 += x - span1->x;
                if(*(span2->covers) == cover_full)
                {
                    sl.add_span(x, len, cover_full);
                }
                else
                {
                    do
                    {
                        cover = cover_mask * cover_mask - 
                                    (cover_mask - *covers1++) * 
                                    (cover_mask - *(span2->covers));
                        sl.add_cell(x++, 
                                    (cover == cover_full * cover_full) ?
                                    cover_full : 
                                    (cover >> cover_shift));
                    }
                    while(--len);
                }
                break;

            case 3:      // Both are solid spans
                cover = cover_mask * cover_mask - 
                            (cover_mask - *(span1->covers)) * 
                            (cover_mask - *(span2->covers));
                sl.add_span(x, len, 
                            (cover == cover_full * cover_full) ?
                            cover_full : 
                            (cover >> cover_shift));
                break;
            }
        }
    };


    //---------------------------------------------sbool_xor_formula_linear
    template<unsigned CoverShift = cover_shift> 
    struct sbool_xor_formula_linear
    {
        enum cover_scale_e
        {
            cover_shift = CoverShift,
            cover_size  = 1 << cover_shift,
            cover_mask  = cover_size - 1
        };

        static AGG_INLINE unsigned calculate(unsigned a, unsigned b)
        {
            unsigned cover = a + b;
            if(cover > cover_mask) cover = cover_mask + cover_mask - cover;
            return cover;
        }
    };


    //---------------------------------------------sbool_xor_formula_saddle
    template<unsigned CoverShift = cover_shift> 
    struct sbool_xor_formula_saddle
    {
        enum cover_scale_e
        {
            cover_shift = CoverShift,
            cover_size  = 1 << cover_shift,
            cover_mask  = cover_size - 1
        };

        static AGG_INLINE unsigned calculate(unsigned a, unsigned b)
        {
            unsigned k = a * b;
            if(k == cover_mask * cover_mask) return 0;

            a = (cover_mask * cover_mask - (a << cover_shift) + k) >> cover_shift;
            b = (cover_mask * cover_mask - (b << cover_shift) + k) >> cover_shift;
            return cover_mask - ((a * b) >> cover_shift);
        }
    };


    //-------------------------------------------sbool_xor_formula_abs_diff
    struct sbool_xor_formula_abs_diff
    {
        static AGG_INLINE unsigned calculate(unsigned a, unsigned b)
        {
            return unsigned(abs(int(a) - int(b)));
        }
    };



    //----------------------------------------------------sbool_xor_spans_aa
    // Functor.
    // XOR two spans preserving the anti-aliasing information.
    // The result is added to the "sl" scanline.
    //------------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class XorFormula,
             unsigned CoverShift = cover_shift> 
    struct sbool_xor_spans_aa
    {
        enum cover_scale_e
        {
            cover_shift = CoverShift,
            cover_size  = 1 << cover_shift,
            cover_mask  = cover_size - 1,
            cover_full  = cover_mask
        };
        

        void operator () (const typename Scanline1::const_iterator& span1, 
                          const typename Scanline2::const_iterator& span2, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            unsigned cover;
            const typename Scanline1::cover_type* covers1;
            const typename Scanline2::cover_type* covers2;

            // Calculate the operation code and choose the 
            // proper combination algorithm.
            // 0 = Both spans are of AA type
            // 1 = span1 is solid, span2 is AA
            // 2 = span1 is AA, span2 is solid
            // 3 = Both spans are of solid type
            //-----------------
            switch((span1->len < 0) | ((span2->len < 0) << 1))
            {
            case 0:      // Both are AA spans
                covers1 = span1->covers;
                covers2 = span2->covers;
                if(span1->x < x) covers1 += x - span1->x;
                if(span2->x < x) covers2 += x - span2->x;
                do
                {
                    cover = XorFormula::calculate(*covers1++, *covers2++);
                    if(cover) sl.add_cell(x, cover);
                    ++x;
                }
                while(--len);
                break;

            case 1:      // span1 is solid, span2 is AA
                covers2 = span2->covers;
                if(span2->x < x) covers2 += x - span2->x;
                do
                {
                    cover = XorFormula::calculate(*(span1->covers), *covers2++);
                    if(cover) sl.add_cell(x, cover);
                    ++x;
                }
                while(--len);
                break;

            case 2:      // span1 is AA, span2 is solid
                covers1 = span1->covers;
                if(span1->x < x) covers1 += x - span1->x;
                do
                {
                    cover = XorFormula::calculate(*covers1++, *(span2->covers));
                    if(cover) sl.add_cell(x, cover);
                    ++x;
                }
                while(--len);
                break;

            case 3:      // Both are solid spans
                cover = XorFormula::calculate(*(span1->covers), *(span2->covers));
                if(cover) sl.add_span(x, len, cover);
                break;

            }
        }
    };





    //-----------------------------------------------sbool_subtract_spans_aa
    // Functor.
    // Unite two spans preserving the anti-aliasing information.
    // The result is added to the "sl" scanline.
    //------------------
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             unsigned CoverShift = cover_shift> 
    struct sbool_subtract_spans_aa
    {
        enum cover_scale_e
        {
            cover_shift = CoverShift,
            cover_size  = 1 << cover_shift,
            cover_mask  = cover_size - 1,
            cover_full  = cover_mask
        };
        

        void operator () (const typename Scanline1::const_iterator& span1, 
                          const typename Scanline2::const_iterator& span2, 
                          int x, unsigned len, 
                          Scanline& sl) const
        {
            unsigned cover;
            const typename Scanline1::cover_type* covers1;
            const typename Scanline2::cover_type* covers2;

            // Calculate the operation code and choose the 
            // proper combination algorithm.
            // 0 = Both spans are of AA type
            // 1 = span1 is solid, span2 is AA
            // 2 = span1 is AA, span2 is solid
            // 3 = Both spans are of solid type
            //-----------------
            switch((span1->len < 0) | ((span2->len < 0) << 1))
            {
            case 0:      // Both are AA spans
                covers1 = span1->covers;
                covers2 = span2->covers;
                if(span1->x < x) covers1 += x - span1->x;
                if(span2->x < x) covers2 += x - span2->x;
                do
                {
                    cover = *covers1++ * (cover_mask - *covers2++);
                    if(cover)
                    {
                        sl.add_cell(x, 
                                    (cover == cover_full * cover_full) ?
                                    cover_full : 
                                    (cover >> cover_shift));
                    }
                    ++x;
                }
                while(--len);
                break;

            case 1:      // span1 is solid, span2 is AA
                covers2 = span2->covers;
                if(span2->x < x) covers2 += x - span2->x;
                do
                {
                    cover = *(span1->covers) * (cover_mask - *covers2++);
                    if(cover)
                    {
                        sl.add_cell(x, 
                                    (cover == cover_full * cover_full) ?
                                    cover_full : 
                                    (cover >> cover_shift));
                    }
                    ++x;
                }
                while(--len);
                break;

            case 2:      // span1 is AA, span2 is solid
                covers1 = span1->covers;
                if(span1->x < x) covers1 += x - span1->x;
                if(*(span2->covers) != cover_full)
                {
                    do
                    {
                        cover = *covers1++ * (cover_mask - *(span2->covers));
                        if(cover)
                        {
                            sl.add_cell(x, 
                                        (cover == cover_full * cover_full) ?
                                        cover_full : 
                                        (cover >> cover_shift));
                        }
                        ++x;
                    }
                    while(--len);
                }
                break;

            case 3:      // Both are solid spans
                cover = *(span1->covers) * (cover_mask - *(span2->covers));
                if(cover)
                {
                    sl.add_span(x, len, 
                                (cover == cover_full * cover_full) ?
                                cover_full : 
                                (cover >> cover_shift));
                }
                break;
            }
        }
    };






    //--------------------------------------------sbool_add_spans_and_render
    template<class Scanline1, 
             class Scanline, 
             class Renderer, 
             class AddSpanFunctor>
    void sbool_add_spans_and_render(const Scanline1& sl1, 
                                    Scanline& sl, 
                                    Renderer& ren, 
                                    AddSpanFunctor add_span)
    {
        sl.reset_spans();
        typename Scanline1::const_iterator span = sl1.begin();
        unsigned num_spans = sl1.num_spans();
        for(;;)
        {
            add_span(span, span->x, abs((int)span->len), sl);
            if(--num_spans == 0) break;
            ++span;
        }
        sl.finalize(sl1.y());
        ren.render(sl);
    }







    //---------------------------------------------sbool_intersect_scanlines
    // Intersect two scanlines, "sl1" and "sl2" and generate a new "sl" one.
    // The combine_spans functor can be of type sbool_combine_spans_bin or
    // sbool_intersect_spans_aa. First is a general functor to combine
    // two spans without Anti-Aliasing, the second preserves the AA
    // information, but works slower
    //
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class CombineSpansFunctor>
    void sbool_intersect_scanlines(const Scanline1& sl1, 
                                   const Scanline2& sl2, 
                                   Scanline& sl, 
                                   CombineSpansFunctor combine_spans)
    {
        sl.reset_spans();

        unsigned num1 = sl1.num_spans();
        if(num1 == 0) return;

        unsigned num2 = sl2.num_spans();
        if(num2 == 0) return;

        typename Scanline1::const_iterator span1 = sl1.begin();
        typename Scanline2::const_iterator span2 = sl2.begin();

        while(num1 && num2)
        {
            int xb1 = span1->x;
            int xb2 = span2->x;
            int xe1 = xb1 + abs((int)span1->len) - 1;
            int xe2 = xb2 + abs((int)span2->len) - 1;

            // Determine what spans we should advance in the next step
            // The span with the least ending X should be advanced
            // advance_both is just an optimization when we ending 
            // coordinates are the same and we can advance both
            //--------------
            bool advance_span1 = xe1 <  xe2;
            bool advance_both  = xe1 == xe2;

            // Find the intersection of the spans
            // and check if they intersect
            //--------------
            if(xb1 < xb2) xb1 = xb2;
            if(xe1 > xe2) xe1 = xe2;
            if(xb1 <= xe1)
            {
                combine_spans(span1, span2, xb1, xe1 - xb1 + 1, sl);
            }

            // Advance the spans
            //--------------
            if(advance_both)
            {
                --num1;
                --num2;
                if(num1) ++span1;
                if(num2) ++span2;
            }
            else
            {
                if(advance_span1)
                {
                    --num1;
                    if(num1) ++span1;
                }
                else
                {
                    --num2;
                    if(num2) ++span2;
                }
            }
        }
    }








    //------------------------------------------------sbool_intersect_shapes
    // Intersect the scanline shapes. Here the "Scanline Generator" 
    // abstraction is used. ScanlineGen1 and ScanlineGen2 are 
    // the generators, and can be of type rasterizer_scanline_aa<>.
    // There function requires three scanline containers that can be of
    // different types.
    // "sl1" and "sl2" are used to retrieve scanlines from the generators,
    // "sl" is ised as the resulting scanline to render it.
    // The external "sl1" and "sl2" are used only for the sake of
    // optimization and reusing of the scanline objects.
    // the function calls sbool_intersect_scanlines with CombineSpansFunctor 
    // as the last argument. See sbool_intersect_scanlines for details.
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer,
             class CombineSpansFunctor>
    void sbool_intersect_shapes(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                Scanline1& sl1, Scanline2& sl2,
                                Scanline& sl, Renderer& ren, 
                                CombineSpansFunctor combine_spans)
    {
        // Prepare the scanline generators.
        // If anyone of them doesn't contain 
        // any scanlines, then return.
        //-----------------
        if(!sg1.rewind_scanlines()) return;
        if(!sg2.rewind_scanlines()) return;

        // Get the bounding boxes
        //----------------
        rect_i r1(sg1.min_x(), sg1.min_y(), sg1.max_x(), sg1.max_y());
        rect_i r2(sg2.min_x(), sg2.min_y(), sg2.max_x(), sg2.max_y());

        // Calculate the intersection of the bounding 
        // boxes and return if they don't intersect.
        //-----------------
        rect_i ir = intersect_rectangles(r1, r2);
        if(!ir.is_valid()) return;

        // Reset the scanlines and get two first ones
        //-----------------
        sl.reset(ir.x1, ir.x2);
        sl1.reset(sg1.min_x(), sg1.max_x());
        sl2.reset(sg2.min_x(), sg2.max_x());
        if(!sg1.sweep_scanline(sl1)) return;
        if(!sg2.sweep_scanline(sl2)) return;

        ren.prepare();

        // The main loop
        // Here we synchronize the scanlines with 
        // the same Y coordinate, ignoring all other ones.
        // Only scanlines having the same Y-coordinate 
        // are to be combined.
        //-----------------
        for(;;)
        {
            while(sl1.y() < sl2.y())
            {
                if(!sg1.sweep_scanline(sl1)) return;
            }
            while(sl2.y() < sl1.y())
            {
                if(!sg2.sweep_scanline(sl2)) return;
            }

            if(sl1.y() == sl2.y())
            {
                // The Y coordinates are the same.
                // Combine the scanlines, render if they contain any spans,
                // and advance both generators to the next scanlines
                //----------------------
                sbool_intersect_scanlines(sl1, sl2, sl, combine_spans);
                if(sl.num_spans())
                {
                    sl.finalize(sl1.y());
                    ren.render(sl);
                }
                if(!sg1.sweep_scanline(sl1)) return;
                if(!sg2.sweep_scanline(sl2)) return;
            }
        }
    }







    //-------------------------------------------------sbool_unite_scanlines
    // Unite two scanlines, "sl1" and "sl2" and generate a new "sl" one.
    // The combine_spans functor can be of type sbool_combine_spans_bin or
    // sbool_intersect_spans_aa. First is a general functor to combine
    // two spans without Anti-Aliasing, the second preserves the AA
    // information, but works slower
    //
    template<class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class AddSpanFunctor1,
             class AddSpanFunctor2,
             class CombineSpansFunctor>
    void sbool_unite_scanlines(const Scanline1& sl1, 
                               const Scanline2& sl2, 
                               Scanline& sl, 
                               AddSpanFunctor1 add_span1,
                               AddSpanFunctor2 add_span2,
                               CombineSpansFunctor combine_spans)
    {
        sl.reset_spans();

        unsigned num1 = sl1.num_spans();
        unsigned num2 = sl2.num_spans();

        typename Scanline1::const_iterator span1;// = sl1.begin();
        typename Scanline2::const_iterator span2;// = sl2.begin();

        enum invalidation_e 
        { 
            invalid_b = 0xFFFFFFF, 
            invalid_e = invalid_b - 1 
        };

        // Initialize the spans as invalid
        //---------------
        int xb1 = invalid_b;
        int xb2 = invalid_b;
        int xe1 = invalid_e;
        int xe2 = invalid_e;

        // Initialize span1 if there are spans
        //---------------
        if(num1)
        {
            span1 = sl1.begin();
            xb1 = span1->x;
            xe1 = xb1 + abs((int)span1->len) - 1;
            --num1;
        }

        // Initialize span2 if there are spans
        //---------------
        if(num2)
        {
            span2 = sl2.begin();
            xb2 = span2->x;
            xe2 = xb2 + abs((int)span2->len) - 1;
            --num2;
        }


        for(;;)
        {
            // Retrieve a new span1 if it's invalid
            //----------------
            if(num1 && xb1 > xe1) 
            {
                --num1;
                ++span1;
                xb1 = span1->x;
                xe1 = xb1 + abs((int)span1->len) - 1;
            }

            // Retrieve a new span2 if it's invalid
            //----------------
            if(num2 && xb2 > xe2) 
            {
                --num2;
                ++span2;
                xb2 = span2->x;
                xe2 = xb2 + abs((int)span2->len) - 1;
            }

            if(xb1 > xe1 && xb2 > xe2) break;

            // Calculate the intersection
            //----------------
            int xb = xb1;
            int xe = xe1;
            if(xb < xb2) xb = xb2;
            if(xe > xe2) xe = xe2;
            int len = xe - xb + 1; // The length of the intersection
            if(len > 0)
            {
                // The spans intersect,
                // add the beginning of the span
                //----------------
                if(xb1 < xb2)
                {
                    add_span1(span1, xb1, xb2 - xb1, sl);
                    xb1 = xb2;
                }
                else
                if(xb2 < xb1)
                {
                    add_span2(span2, xb2, xb1 - xb2, sl);
                    xb2 = xb1;
                }

                // Add the combination part of the spans
                //----------------
                combine_spans(span1, span2, xb, len, sl);


                // Invalidate the fully processed span or both
                //----------------
                if(xe1 < xe2)
                {
                    // Invalidate span1 and eat
                    // the processed part of span2
                    //--------------
                    xb1 = invalid_b;    
                    xe1 = invalid_e;
                    xb2 += len;
                }
                else
                if(xe2 < xe1)
                {
                    // Invalidate span2 and eat
                    // the processed part of span1
                    //--------------
                    xb2 = invalid_b;  
                    xe2 = invalid_e;
                    xb1 += len;
                }
                else
                {
                    xb1 = invalid_b;  // Invalidate both
                    xb2 = invalid_b;
                    xe1 = invalid_e;
                    xe2 = invalid_e;
                }
            }
            else
            {
                // The spans do not intersect
                //--------------
                if(xb1 < xb2) 
                {
                    // Advance span1
                    //---------------
                    if(xb1 <= xe1)
                    {
                        add_span1(span1, xb1, xe1 - xb1 + 1, sl);
                    }
                    xb1 = invalid_b; // Invalidate
                    xe1 = invalid_e;
                }
                else
                {
                    // Advance span2
                    //---------------
                    if(xb2 <= xe2)
                    {
                        add_span2(span2, xb2, xe2 - xb2 + 1, sl);
                    }
                    xb2 = invalid_b; // Invalidate
                    xe2 = invalid_e;
                }
            }
        }
    }




    //----------------------------------------------------sbool_unite_shapes
    // Unite the scanline shapes. Here the "Scanline Generator" 
    // abstraction is used. ScanlineGen1 and ScanlineGen2 are 
    // the generators, and can be of type rasterizer_scanline_aa<>.
    // There function requires three scanline containers that can be 
    // of different type.
    // "sl1" and "sl2" are used to retrieve scanlines from the generators,
    // "sl" is ised as the resulting scanline to render it.
    // The external "sl1" and "sl2" are used only for the sake of
    // optimization and reusing of the scanline objects.
    // the function calls sbool_unite_scanlines with CombineSpansFunctor 
    // as the last argument. See sbool_unite_scanlines for details.
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer,
             class AddSpanFunctor1,
             class AddSpanFunctor2,
             class CombineSpansFunctor>
    void sbool_unite_shapes(ScanlineGen1& sg1, ScanlineGen2& sg2,
                            Scanline1& sl1, Scanline2& sl2,
                            Scanline& sl, Renderer& ren, 
                            AddSpanFunctor1 add_span1,
                            AddSpanFunctor2 add_span2,
                            CombineSpansFunctor combine_spans)
    {
        // Prepare the scanline generators.
        // If anyone of them doesn't contain 
        // any scanlines, then return.
        //-----------------
        bool flag1 = sg1.rewind_scanlines();
        bool flag2 = sg2.rewind_scanlines();
        if(!flag1 && !flag2) return;

        // Get the bounding boxes
        //----------------
        rect_i r1(sg1.min_x(), sg1.min_y(), sg1.max_x(), sg1.max_y());
        rect_i r2(sg2.min_x(), sg2.min_y(), sg2.max_x(), sg2.max_y());

        // Calculate the union of the bounding boxes
        //-----------------
        rect_i ur(1,1,0,0);
             if(flag1 && flag2) ur = unite_rectangles(r1, r2);
        else if(flag1)          ur = r1;
        else if(flag2)          ur = r2;

        if(!ur.is_valid()) return;

        ren.prepare();

        // Reset the scanlines and get two first ones
        //-----------------
        sl.reset(ur.x1, ur.x2);
        if(flag1) 
        {
            sl1.reset(sg1.min_x(), sg1.max_x());
            flag1 = sg1.sweep_scanline(sl1);
        }

        if(flag2) 
        {
            sl2.reset(sg2.min_x(), sg2.max_x());
            flag2 = sg2.sweep_scanline(sl2);
        }

        // The main loop
        // Here we synchronize the scanlines with 
        // the same Y coordinate.
        //-----------------
        while(flag1 || flag2)
        {
            if(flag1 && flag2)
            {
                if(sl1.y() == sl2.y())
                {
                    // The Y coordinates are the same.
                    // Combine the scanlines, render if they contain any spans,
                    // and advance both generators to the next scanlines
                    //----------------------
                    sbool_unite_scanlines(sl1, sl2, sl, 
                                          add_span1, add_span2, combine_spans);
                    if(sl.num_spans())
                    {
                        sl.finalize(sl1.y());
                        ren.render(sl);
                    }
                    flag1 = sg1.sweep_scanline(sl1);
                    flag2 = sg2.sweep_scanline(sl2);
                }
                else
                {
                    if(sl1.y() < sl2.y())
                    {
                        sbool_add_spans_and_render(sl1, sl, ren, add_span1);
                        flag1 = sg1.sweep_scanline(sl1);
                    }
                    else
                    {
                        sbool_add_spans_and_render(sl2, sl, ren, add_span2);
                        flag2 = sg2.sweep_scanline(sl2);
                    }
                }
            }
            else
            {
                if(flag1)
                {
                    sbool_add_spans_and_render(sl1, sl, ren, add_span1);
                    flag1 = sg1.sweep_scanline(sl1);
                }
                if(flag2)
                {
                    sbool_add_spans_and_render(sl2, sl, ren, add_span2);
                    flag2 = sg2.sweep_scanline(sl2);
                }
            }
        }
    }








    //-------------------------------------------------sbool_subtract_shapes
    // Subtract the scanline shapes, "sg1-sg2". Here the "Scanline Generator" 
    // abstraction is used. ScanlineGen1 and ScanlineGen2 are 
    // the generators, and can be of type rasterizer_scanline_aa<>.
    // There function requires three scanline containers that can be of
    // different types.
    // "sl1" and "sl2" are used to retrieve scanlines from the generators,
    // "sl" is ised as the resulting scanline to render it.
    // The external "sl1" and "sl2" are used only for the sake of
    // optimization and reusing of the scanline objects.
    // the function calls sbool_intersect_scanlines with CombineSpansFunctor 
    // as the last argument. See combine_scanlines_sub for details.
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer,
             class AddSpanFunctor1,
             class CombineSpansFunctor>
    void sbool_subtract_shapes(ScanlineGen1& sg1, ScanlineGen2& sg2,
                               Scanline1& sl1, Scanline2& sl2,
                               Scanline& sl, Renderer& ren, 
                               AddSpanFunctor1 add_span1,
                               CombineSpansFunctor combine_spans)
    {
        // Prepare the scanline generators.
        // Here "sg1" is master, "sg2" is slave.
        //-----------------
        if(!sg1.rewind_scanlines()) return;
        bool flag2 = sg2.rewind_scanlines();

        // Get the bounding box
        //----------------
        rect_i r1(sg1.min_x(), sg1.min_y(), sg1.max_x(), sg1.max_y());

        // Reset the scanlines and get two first ones
        //-----------------
        sl.reset(sg1.min_x(), sg1.max_x());
        sl1.reset(sg1.min_x(), sg1.max_x());
        sl2.reset(sg2.min_x(), sg2.max_x());
        if(!sg1.sweep_scanline(sl1)) return;

        if(flag2) flag2 = sg2.sweep_scanline(sl2);

        ren.prepare();

        // A fake span2 processor
        sbool_add_span_empty<Scanline2, Scanline> add_span2;

        // The main loop
        // Here we synchronize the scanlines with 
        // the same Y coordinate, ignoring all other ones.
        // Only scanlines having the same Y-coordinate 
        // are to be combined.
        //-----------------
        bool flag1 = true;
        do
        {
            // Synchronize "slave" with "master"
            //-----------------
            while(flag2 && sl2.y() < sl1.y())
            {
                flag2 = sg2.sweep_scanline(sl2);
            }


            if(flag2 && sl2.y() == sl1.y())
            {
                // The Y coordinates are the same.
                // Combine the scanlines and render if they contain any spans.
                //----------------------
                sbool_unite_scanlines(sl1, sl2, sl, add_span1, add_span2, combine_spans);
                if(sl.num_spans())
                {
                    sl.finalize(sl1.y());
                    ren.render(sl);
                }
            }
            else
            {
                sbool_add_spans_and_render(sl1, sl, ren, add_span1);
            }

            // Advance the "master"
            flag1 = sg1.sweep_scanline(sl1);
        }
        while(flag1);
    }







    //---------------------------------------------sbool_intersect_shapes_aa
    // Intersect two anti-aliased scanline shapes. 
    // Here the "Scanline Generator" abstraction is used. 
    // ScanlineGen1 and ScanlineGen2 are the generators, and can be of 
    // type rasterizer_scanline_aa<>. There function requires three 
    // scanline containers that can be of different types.
    // "sl1" and "sl2" are used to retrieve scanlines from the generators,
    // "sl" is ised as the resulting scanline to render it.
    // The external "sl1" and "sl2" are used only for the sake of
    // optimization and reusing of the scanline objects.
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_intersect_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                   Scanline1& sl1, Scanline2& sl2,
                                   Scanline& sl, Renderer& ren)
    {
        sbool_intersect_spans_aa<Scanline1, Scanline2, Scanline> combine_functor;
        sbool_intersect_shapes(sg1, sg2, sl1, sl2, sl, ren, combine_functor);
    }





    //--------------------------------------------sbool_intersect_shapes_bin
    // Intersect two binary scanline shapes (without anti-aliasing). 
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_intersect_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                    Scanline1& sl1, Scanline2& sl2,
                                    Scanline& sl, Renderer& ren)
    {
        sbool_combine_spans_bin<Scanline1, Scanline2, Scanline> combine_functor;
        sbool_intersect_shapes(sg1, sg2, sl1, sl2, sl, ren, combine_functor);
    }





    //-------------------------------------------------sbool_unite_shapes_aa
    // Unite two anti-aliased scanline shapes 
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_unite_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                               Scanline1& sl1, Scanline2& sl2,
                               Scanline& sl, Renderer& ren)
    {
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        sbool_unite_spans_aa<Scanline1, Scanline2, Scanline> combine_functor;
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }





    //------------------------------------------------sbool_unite_shapes_bin
    // Unite two binary scanline shapes (without anti-aliasing). 
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_unite_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                Scanline1& sl1, Scanline2& sl2,
                                Scanline& sl, Renderer& ren)
    {
        sbool_add_span_bin<Scanline1, Scanline> add_functor1;
        sbool_add_span_bin<Scanline2, Scanline> add_functor2;
        sbool_combine_spans_bin<Scanline1, Scanline2, Scanline> combine_functor;
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }









    //---------------------------------------------------sbool_xor_shapes_aa
    // Apply eXclusive OR to two anti-aliased scanline shapes. There's 
    // a modified "Linear" XOR used instead of classical "Saddle" one.
    // The reason is to have the result absolutely conststent with what
    // the scanline rasterizer produces.
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                             Scanline1& sl1, Scanline2& sl2,
                             Scanline& sl, Renderer& ren)
    {
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        sbool_xor_spans_aa<Scanline1, Scanline2, Scanline, 
                           sbool_xor_formula_linear<> > combine_functor;
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }



    //------------------------------------------sbool_xor_shapes_saddle_aa
    // Apply eXclusive OR to two anti-aliased scanline shapes. 
    // There's the classical "Saddle" used to calculate the 
    // Anti-Aliasing values, that is:
    // a XOR b : 1-((1-a+a*b)*(1-b+a*b))
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_saddle_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                    Scanline1& sl1, Scanline2& sl2,
                                    Scanline& sl, Renderer& ren)
    {
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        sbool_xor_spans_aa<Scanline1, 
                           Scanline2, 
                           Scanline, 
                           sbool_xor_formula_saddle<> > combine_functor;
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }


    //--------------------------------------sbool_xor_shapes_abs_diff_aa
    // Apply eXclusive OR to two anti-aliased scanline shapes. 
    // There's the absolute difference used to calculate 
    // Anti-Aliasing values, that is:
    // a XOR b : abs(a-b)
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_abs_diff_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                      Scanline1& sl1, Scanline2& sl2,
                                      Scanline& sl, Renderer& ren)
    {
        sbool_add_span_aa<Scanline1, Scanline> add_functor1;
        sbool_add_span_aa<Scanline2, Scanline> add_functor2;
        sbool_xor_spans_aa<Scanline1, 
                           Scanline2, 
                           Scanline, 
                           sbool_xor_formula_abs_diff> combine_functor;
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }



    //--------------------------------------------------sbool_xor_shapes_bin
    // Apply eXclusive OR to two binary scanline shapes (without anti-aliasing). 
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_xor_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                              Scanline1& sl1, Scanline2& sl2,
                              Scanline& sl, Renderer& ren)
    {
        sbool_add_span_bin<Scanline1, Scanline> add_functor1;
        sbool_add_span_bin<Scanline2, Scanline> add_functor2;
        sbool_combine_spans_empty<Scanline1, Scanline2, Scanline> combine_functor;
        sbool_unite_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                           add_functor1, add_functor2, combine_functor);
    }






    //----------------------------------------------sbool_subtract_shapes_aa
    // Subtract shapes "sg1-sg2" with anti-aliasing
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_subtract_shapes_aa(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                  Scanline1& sl1, Scanline2& sl2,
                                  Scanline& sl, Renderer& ren)
    {
        sbool_add_span_aa<Scanline1, Scanline> add_functor;
        sbool_subtract_spans_aa<Scanline1, Scanline2, Scanline> combine_functor;
        sbool_subtract_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                              add_functor, combine_functor);
    }





    //---------------------------------------------sbool_subtract_shapes_bin
    // Subtract binary shapes "sg1-sg2" without anti-aliasing
    // See intersect_shapes_aa for more comments
    //----------
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_subtract_shapes_bin(ScanlineGen1& sg1, ScanlineGen2& sg2,
                                   Scanline1& sl1, Scanline2& sl2,
                                   Scanline& sl, Renderer& ren)
    {
        sbool_add_span_bin<Scanline1, Scanline> add_functor;
        sbool_combine_spans_empty<Scanline1, Scanline2, Scanline> combine_functor;
        sbool_subtract_shapes(sg1, sg2, sl1, sl2, sl, ren, 
                              add_functor, combine_functor);
    }






    //------------------------------------------------------------sbool_op_e
    enum sbool_op_e
    {
        sbool_or,            //----sbool_or
        sbool_and,           //----sbool_and
        sbool_xor,           //----sbool_xor
        sbool_xor_saddle,    //----sbool_xor_saddle
        sbool_xor_abs_diff,  //----sbool_xor_abs_diff
        sbool_a_minus_b,     //----sbool_a_minus_b
        sbool_b_minus_a      //----sbool_b_minus_a
    };






    //----------------------------------------------sbool_combine_shapes_bin
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_combine_shapes_bin(sbool_op_e op,
                                  ScanlineGen1& sg1, ScanlineGen2& sg2,
                                  Scanline1& sl1, Scanline2& sl2,
                                  Scanline& sl, Renderer& ren)
    {
        switch(op)
        {
        case sbool_or          : sbool_unite_shapes_bin    (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_and         : sbool_intersect_shapes_bin(sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor         :
        case sbool_xor_saddle  : 
        case sbool_xor_abs_diff: sbool_xor_shapes_bin      (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_a_minus_b   : sbool_subtract_shapes_bin (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_b_minus_a   : sbool_subtract_shapes_bin (sg2, sg1, sl2, sl1, sl, ren); break;
        }
    }




    //-----------------------------------------------sbool_combine_shapes_aa
    template<class ScanlineGen1, 
             class ScanlineGen2, 
             class Scanline1, 
             class Scanline2, 
             class Scanline, 
             class Renderer>
    void sbool_combine_shapes_aa(sbool_op_e op,
                                 ScanlineGen1& sg1, ScanlineGen2& sg2,
                                 Scanline1& sl1, Scanline2& sl2,
                                 Scanline& sl, Renderer& ren)
    {
        switch(op)
        {
        case sbool_or          : sbool_unite_shapes_aa       (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_and         : sbool_intersect_shapes_aa   (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor         : sbool_xor_shapes_aa         (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor_saddle  : sbool_xor_shapes_saddle_aa  (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_xor_abs_diff: sbool_xor_shapes_abs_diff_aa(sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_a_minus_b   : sbool_subtract_shapes_aa    (sg1, sg2, sl1, sl2, sl, ren); break;
        case sbool_b_minus_a   : sbool_subtract_shapes_aa    (sg2, sg1, sl2, sl1, sl, ren); break;
        }
    }

}


#endif

