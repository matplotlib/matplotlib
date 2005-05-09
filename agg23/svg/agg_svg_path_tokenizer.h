//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.3
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
//
// SVG path tokenizer.
//
//----------------------------------------------------------------------------
#ifndef AGG_SVG_PATH_TOKENIZER_INCLUDED
#define AGG_SVG_PATH_TOKENIZER_INCLUDED

#include "agg_svg_exception.h"

namespace agg 
{ 
namespace svg
{
    // SVG path tokenizer. 
    // Example:
    //
    // agg::svg::path_tokenizer tok;
    //
    // tok.set_str("M-122.304 84.285L-122.304 84.285 122.203 86.179 ");
    // while(tok.next())
    // {
    //     printf("command='%c' number=%f\n", 
    //             tok.last_command(), 
    //             tok.last_number());
    // }
    //
    // The tokenizer does all the routine job of parsing the SVG paths.
    // It doesn't recognize any graphical primitives, it even doesn't know
    // anything about pairs of coordinates (X,Y). The purpose of this class 
    // is to tokenize the numeric values and commands. SVG paths can 
    // have single numeric values for Horizontal or Vertical line_to commands 
    // as well as more than two coordinates (4 or 6) for Bezier curves 
    // depending on the semantics of the command.
    // The behaviour is as follows:
    //
    // Each call to next() returns true if there's new command or new numeric
    // value or false when the path ends. How to interpret the result
    // depends on the sematics of the command. For example, command "C" 
    // (cubic Bezier curve) implies 6 floating point numbers preceded by this 
    // command. If the command assumes no arguments (like z or Z) the 
    // the last_number() values won't change, that is, last_number() always
    // returns the last recognized numeric value, so does last_command().
    //===============================================================
    class path_tokenizer
    {
     public:
        path_tokenizer();

        void set_path_str(const char* str);
        bool next();

        double next(char cmd);

        char   last_command() const { return m_last_command; }
        double last_number() const { return m_last_number; }


    private:
        static void init_char_mask(char* mask, const char* char_set);

        bool contains(const char* mask, unsigned c) const
        {
            return (mask[(c >> 3) & (256/8-1)] & (1 << (c & 7))) != 0;
        }

        bool is_command(unsigned c) const
        {
            return contains(m_commands_mask, c);
        }

        bool is_numeric(unsigned c) const
        {
            return contains(m_numeric_mask, c);
        }

        bool is_separator(unsigned c) const
        {
            return contains(m_separators_mask, c);
        }

        bool parse_number();

        char m_separators_mask[256/8];
        char m_commands_mask[256/8];
        char m_numeric_mask[256/8];

        const char* m_path;
        double m_last_number;
        char   m_last_command;

        static const char s_commands[];
        static const char s_numeric[];
        static const char s_separators[];
    };

} //namespace svg
} //namespace agg


#endif
