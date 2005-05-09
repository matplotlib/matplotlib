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
// SVG parser.
//
//----------------------------------------------------------------------------

#ifndef AGG_SVG_PARSER_INCLUDED
#define AGG_SVG_PARSER_INCLUDED

#include "agg_svg_path_tokenizer.h"
#include "agg_svg_path_renderer.h"

namespace agg
{
namespace svg
{

    class parser
    {
        enum { buf_size = BUFSIZ };
    public:

        ~parser();
        parser(path_renderer& path);

        void parse(const char* fname);
        const char* title() const { return m_title; }

    private:
        // XML event handlers
        static void start_element(void* data, const char* el, const char** attr);
        static void end_element(void* data, const char* el);
        static void content(void* data, const char* s, int len);

        void parse_attr(const char** attr);
        void parse_path(const char** attr);
        void parse_poly(const char** attr, bool close_flag);
        void parse_rect(const char** attr);
        void parse_line(const char** attr);
        void parse_style(const char* str);
        void parse_transform(const char* str);

        unsigned parse_matrix(const char* str);
        unsigned parse_translate(const char* str);
        unsigned parse_rotate(const char* str);
        unsigned parse_scale(const char* str);
        unsigned parse_skew_x(const char* str);
        unsigned parse_skew_y(const char* str);
        
        bool parse_attr(const char* name, const char* value);
        bool parse_name_value(const char* nv_start, const char* nv_end);
        void copy_name(const char* start, const char* end);
        void copy_value(const char* start, const char* end);
        
    private:
        path_renderer& m_path;
        path_tokenizer m_tokenizer;
        char*          m_buf;
        char*          m_title;
        unsigned       m_title_len;
        bool           m_title_flag;
        bool           m_path_flag;
        char*          m_attr_name;
        char*          m_attr_value;
        unsigned       m_attr_name_len;
        unsigned       m_attr_value_len;
    };

}
}

#endif
