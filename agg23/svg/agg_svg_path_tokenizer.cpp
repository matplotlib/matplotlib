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
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "agg_svg_exception.h"
#include "agg_svg_path_tokenizer.h"


namespace agg 
{ 
namespace svg
{

    //------------------------------------------------------------------------
    const char path_tokenizer::s_commands[]   = "+-MmZzLlHhVvCcSsQqTtAaFfPp";
    const char path_tokenizer::s_numeric[]    = ".Ee0123456789";
    const char path_tokenizer::s_separators[] = " ,\t\n\r";

    //------------------------------------------------------------------------
    path_tokenizer::path_tokenizer()
        : m_path(0), m_last_command(0), m_last_number(0.0)
    {
        init_char_mask(m_commands_mask,   s_commands);
        init_char_mask(m_numeric_mask,    s_numeric);
        init_char_mask(m_separators_mask, s_separators);
    }


    //------------------------------------------------------------------------
    void path_tokenizer::set_path_str(const char* str)
    {
        m_path = str;
        m_last_command = 0;
        m_last_number = 0.0;
    }


    //------------------------------------------------------------------------
    void path_tokenizer::init_char_mask(char* mask, const char* char_set)
    {
        memset(mask, 0, 256/8);
        while(*char_set) 
        {
            unsigned c = unsigned(*char_set++) & 0xFF;
            mask[c >> 3] |= 1 << (c & 7);
        }
    }


    //------------------------------------------------------------------------
    bool path_tokenizer::next()
    {
        if(m_path == 0) return false;

        // Skip all white spaces and other garbage
        while(*m_path && !is_command(*m_path) && !is_numeric(*m_path)) 
        {
            if(!is_separator(*m_path))
            {
                char buf[100];
                sprintf(buf, "path_tokenizer::next : Invalid Character %c", *m_path);
                throw exception(buf);
            }
            m_path++;
        }

        if(*m_path == 0) return false;

        if(is_command(*m_path))
        {
            // Check if the command is a numeric sign character
            if(*m_path == '-' || *m_path == '+')
            {
                return parse_number();
            }
            m_last_command = *m_path++;
            while(*m_path && is_separator(*m_path)) m_path++;
            if(*m_path == 0) return true;
        }
        return parse_number();
    }



    //------------------------------------------------------------------------
    double path_tokenizer::next(char cmd)
    {
        if(!next()) throw exception("parse_path: Unexpected end of path");
        if(last_command() != cmd)
        {
            char buf[100];
            sprintf(buf, "parse_path: Command %c: bad or missing parameters", cmd);
            throw exception(buf);
        }
        return last_number();
    }


    //------------------------------------------------------------------------
    bool path_tokenizer::parse_number()
    {
        char buf[256]; // Should be enough for any number
        char* buf_ptr = buf;

        // Copy all sign characters
        while(buf_ptr < buf+255 && *m_path == '-' || *m_path == '+')
        {
            *buf_ptr++ = *m_path++;
        }

        // Copy all numeric characters
        while(buf_ptr < buf+255 && is_numeric(*m_path))
        {
            *buf_ptr++ = *m_path++;
        }
        *buf_ptr = 0;
        m_last_number = atof(buf);
        return true;
    }


} //namespace svg
} //namespace agg




