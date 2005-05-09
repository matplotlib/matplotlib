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

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "agg_svg_parser.h"
#include "expat.h"

namespace agg
{
namespace svg
{
    struct named_color
    {
        char  name[22];
        int8u r, g, b, a;
    };

    named_color colors[] = 
    {
        { "aliceblue",240,248,255, 255 },
        { "antiquewhite",250,235,215, 255 },
        { "aqua",0,255,255, 255 },
        { "aquamarine",127,255,212, 255 },
        { "azure",240,255,255, 255 },
        { "beige",245,245,220, 255 },
        { "bisque",255,228,196, 255 },
        { "black",0,0,0, 255 },
        { "blanchedalmond",255,235,205, 255 },
        { "blue",0,0,255, 255 },
        { "blueviolet",138,43,226, 255 },
        { "brown",165,42,42, 255 },
        { "burlywood",222,184,135, 255 },
        { "cadetblue",95,158,160, 255 },
        { "chartreuse",127,255,0, 255 },
        { "chocolate",210,105,30, 255 },
        { "coral",255,127,80, 255 },
        { "cornflowerblue",100,149,237, 255 },
        { "cornsilk",255,248,220, 255 },
        { "crimson",220,20,60, 255 },
        { "cyan",0,255,255, 255 },
        { "darkblue",0,0,139, 255 },
        { "darkcyan",0,139,139, 255 },
        { "darkgoldenrod",184,134,11, 255 },
        { "darkgray",169,169,169, 255 },
        { "darkgreen",0,100,0, 255 },
        { "darkgrey",169,169,169, 255 },
        { "darkkhaki",189,183,107, 255 },
        { "darkmagenta",139,0,139, 255 },
        { "darkolivegreen",85,107,47, 255 },
        { "darkorange",255,140,0, 255 },
        { "darkorchid",153,50,204, 255 },
        { "darkred",139,0,0, 255 },
        { "darksalmon",233,150,122, 255 },
        { "darkseagreen",143,188,143, 255 },
        { "darkslateblue",72,61,139, 255 },
        { "darkslategray",47,79,79, 255 },
        { "darkslategrey",47,79,79, 255 },
        { "darkturquoise",0,206,209, 255 },
        { "darkviolet",148,0,211, 255 },
        { "deeppink",255,20,147, 255 },
        { "deepskyblue",0,191,255, 255 },
        { "dimgray",105,105,105, 255 },
        { "dimgrey",105,105,105, 255 },
        { "dodgerblue",30,144,255, 255 },
        { "firebrick",178,34,34, 255 },
        { "floralwhite",255,250,240, 255 },
        { "forestgreen",34,139,34, 255 },
        { "fuchsia",255,0,255, 255 },
        { "gainsboro",220,220,220, 255 },
        { "ghostwhite",248,248,255, 255 },
        { "gold",255,215,0, 255 },
        { "goldenrod",218,165,32, 255 },
        { "gray",128,128,128, 255 },
        { "green",0,128,0, 255 },
        { "greenyellow",173,255,47, 255 },
        { "grey",128,128,128, 255 },
        { "honeydew",240,255,240, 255 },
        { "hotpink",255,105,180, 255 },
        { "indianred",205,92,92, 255 },
        { "indigo",75,0,130, 255 },
        { "ivory",255,255,240, 255 },
        { "khaki",240,230,140, 255 },
        { "lavender",230,230,250, 255 },
        { "lavenderblush",255,240,245, 255 },
        { "lawngreen",124,252,0, 255 },
        { "lemonchiffon",255,250,205, 255 },
        { "lightblue",173,216,230, 255 },
        { "lightcoral",240,128,128, 255 },
        { "lightcyan",224,255,255, 255 },
        { "lightgoldenrodyellow",250,250,210, 255 },
        { "lightgray",211,211,211, 255 },
        { "lightgreen",144,238,144, 255 },
        { "lightgrey",211,211,211, 255 },
        { "lightpink",255,182,193, 255 },
        { "lightsalmon",255,160,122, 255 },
        { "lightseagreen",32,178,170, 255 },
        { "lightskyblue",135,206,250, 255 },
        { "lightslategray",119,136,153, 255 },
        { "lightslategrey",119,136,153, 255 },
        { "lightsteelblue",176,196,222, 255 },
        { "lightyellow",255,255,224, 255 },
        { "lime",0,255,0, 255 },
        { "limegreen",50,205,50, 255 },
        { "linen",250,240,230, 255 },
        { "magenta",255,0,255, 255 },
        { "maroon",128,0,0, 255 },
        { "mediumaquamarine",102,205,170, 255 },
        { "mediumblue",0,0,205, 255 },
        { "mediumorchid",186,85,211, 255 },
        { "mediumpurple",147,112,219, 255 },
        { "mediumseagreen",60,179,113, 255 },
        { "mediumslateblue",123,104,238, 255 },
        { "mediumspringgreen",0,250,154, 255 },
        { "mediumturquoise",72,209,204, 255 },
        { "mediumvioletred",199,21,133, 255 },
        { "midnightblue",25,25,112, 255 },
        { "mintcream",245,255,250, 255 },
        { "mistyrose",255,228,225, 255 },
        { "moccasin",255,228,181, 255 },
        { "navajowhite",255,222,173, 255 },
        { "navy",0,0,128, 255 },
        { "oldlace",253,245,230, 255 },
        { "olive",128,128,0, 255 },
        { "olivedrab",107,142,35, 255 },
        { "orange",255,165,0, 255 },
        { "orangered",255,69,0, 255 },
        { "orchid",218,112,214, 255 },
        { "palegoldenrod",238,232,170, 255 },
        { "palegreen",152,251,152, 255 },
        { "paleturquoise",175,238,238, 255 },
        { "palevioletred",219,112,147, 255 },
        { "papayawhip",255,239,213, 255 },
        { "peachpuff",255,218,185, 255 },
        { "peru",205,133,63, 255 },
        { "pink",255,192,203, 255 },
        { "plum",221,160,221, 255 },
        { "powderblue",176,224,230, 255 },
        { "purple",128,0,128, 255 },
        { "red",255,0,0, 255 },
        { "rosybrown",188,143,143, 255 },
        { "royalblue",65,105,225, 255 },
        { "saddlebrown",139,69,19, 255 },
        { "salmon",250,128,114, 255 },
        { "sandybrown",244,164,96, 255 },
        { "seagreen",46,139,87, 255 },
        { "seashell",255,245,238, 255 },
        { "sienna",160,82,45, 255 },
        { "silver",192,192,192, 255 },
        { "skyblue",135,206,235, 255 },
        { "slateblue",106,90,205, 255 },
        { "slategray",112,128,144, 255 },
        { "slategrey",112,128,144, 255 },
        { "snow",255,250,250, 255 },
        { "springgreen",0,255,127, 255 },
        { "steelblue",70,130,180, 255 },
        { "tan",210,180,140, 255 },
        { "teal",0,128,128, 255 },
        { "thistle",216,191,216, 255 },
        { "tomato",255,99,71, 255 },
        { "turquoise",64,224,208, 255 },
        { "violet",238,130,238, 255 },
        { "wheat",245,222,179, 255 },
        { "white",255,255,255, 255 },
        { "whitesmoke",245,245,245, 255 },
        { "yellow",255,255,0, 255 },
        { "yellowgreen",154,205,50, 255 },
        { "zzzzzzzzzzz",0,0,0, 0 }
    }; 


    //------------------------------------------------------------------------
    parser::~parser()
    {
        delete [] m_attr_value;
        delete [] m_attr_name;
        delete [] m_buf;
        delete [] m_title;
    }

    //------------------------------------------------------------------------
    parser::parser(path_renderer& path) :
        m_path(path),
        m_tokenizer(),
        m_buf(new char[buf_size]),
        m_title(new char[256]),
        m_title_len(0),
        m_title_flag(false),
        m_path_flag(false),
        m_attr_name(new char[128]),
        m_attr_value(new char[1024]),
        m_attr_name_len(127),
        m_attr_value_len(1023)
    {
        m_title[0] = 0;
    }

    //------------------------------------------------------------------------
    void parser::parse(const char* fname)
    {
        char msg[1024];
	    XML_Parser p = XML_ParserCreate(NULL);
	    if(p == 0) 
	    {
		    throw exception("Couldn't allocate memory for parser");
	    }

        XML_SetUserData(p, this);
	    XML_SetElementHandler(p, start_element, end_element);
	    XML_SetCharacterDataHandler(p, content);

        FILE* fd = fopen(fname, "r");
        if(fd == 0)
        {
            sprintf(msg, "Couldn't open file %s", fname);
		    throw exception(msg);
        }

        bool done = false;
        do
        {
            size_t len = fread(m_buf, 1, buf_size, fd);
            done = len < buf_size;
            if(!XML_Parse(p, m_buf, len, done))
            {
                sprintf(msg,
                    "%s at line %d\n",
                    XML_ErrorString(XML_GetErrorCode(p)),
                    XML_GetCurrentLineNumber(p));
                throw exception(msg);
            }
        }
        while(!done);
        fclose(fd);
        XML_ParserFree(p);

        char* ts = m_title;
        while(*ts)
        {
            if(*ts < ' ') *ts = ' ';
            ++ts;
        }
    }


    //------------------------------------------------------------------------
    void parser::start_element(void* data, const char* el, const char** attr)
    {
        parser& self = *(parser*)data;

        if(strcmp(el, "title") == 0)
        {
            self.m_title_flag = true;
        }
        else
        if(strcmp(el, "g") == 0)
        {
            self.m_path.push_attr();
            self.parse_attr(attr);
        }
        else
        if(strcmp(el, "path") == 0)
        {
            if(self.m_path_flag)
            {
                throw exception("start_element: Nested path");
            }
            self.m_path.begin_path();
            self.parse_path(attr);
            self.m_path.end_path();
            self.m_path_flag = true;
        }
        else
        if(strcmp(el, "rect") == 0) 
        {
            self.parse_rect(attr);
        }
        else
        if(strcmp(el, "line") == 0) 
        {
            self.parse_line(attr);
        }
        else
        if(strcmp(el, "polyline") == 0) 
        {
            self.parse_poly(attr, false);
        }
        else
        if(strcmp(el, "polygon") == 0) 
        {
            self.parse_poly(attr, true);
        }
        //else
        //if(strcmp(el, "<OTHER_ELEMENTS>") == 0) 
        //{
        //}
        // . . .
    } 


    //------------------------------------------------------------------------
    void parser::end_element(void* data, const char* el)
    {
        parser& self = *(parser*)data;

        if(strcmp(el, "title") == 0)
        {
            self.m_title_flag = false;
        }
        else
        if(strcmp(el, "g") == 0)
        {
            self.m_path.pop_attr();
        }
        else
        if(strcmp(el, "path") == 0)
        {
            self.m_path_flag = false;
        }
        //else
        //if(strcmp(el, "<OTHER_ELEMENTS>") == 0) 
        //{
        //}
        // . . .
    }


    //------------------------------------------------------------------------
    void parser::content(void* data, const char* s, int len)
    {
        parser& self = *(parser*)data;

        // m_title_flag signals that the <title> tag is being parsed now.
        // The following code concatenates the pieces of content of the <title> tag.
        if(self.m_title_flag)
        {
            if(len + self.m_title_len > 255) len = 255 - self.m_title_len;
            if(len > 0) 
            {
                memcpy(self.m_title + self.m_title_len, s, len);
                self.m_title_len += len;
                self.m_title[self.m_title_len] = 0;
            }
        }
    }


    //------------------------------------------------------------------------
    void parser::parse_attr(const char** attr)
    {
        int i;
        for(i = 0; attr[i]; i += 2)
        {
            if(strcmp(attr[i], "style") == 0)
            {
                parse_style(attr[i + 1]);
            }
            else
            {
                parse_attr(attr[i], attr[i + 1]);
            }
        }
    }

    //-------------------------------------------------------------
    void parser::parse_path(const char** attr)
    {
        int i;

        for(i = 0; attr[i]; i += 2)
        {
            // The <path> tag can consist of the path itself ("d=") 
            // as well as of other parameters like "style=", "transform=", etc.
            // In the last case we simply rely on the function of parsing 
            // attributes (see 'else' branch).
            if(strcmp(attr[i], "d") == 0)
            {
                m_tokenizer.set_path_str(attr[i + 1]);
                m_path.parse_path(m_tokenizer);
            }
            else
            {
                // Create a temporary single pair "name-value" in order
                // to avoid multiple calls for the same attribute.
                const char* tmp[4];
                tmp[0] = attr[i];
                tmp[1] = attr[i + 1];
                tmp[2] = 0;
                tmp[3] = 0;
                parse_attr(tmp);
            }
        }
    }


    //-------------------------------------------------------------
    int cmp_color(const void* p1, const void* p2)
    {
        return strcmp(((named_color*)p1)->name, ((named_color*)p2)->name);
    }

    //-------------------------------------------------------------
    rgba8 parse_color(const char* str)
    {
        while(*str == ' ') ++str;
        unsigned c = 0;
        if(*str == '#')
        {
            sscanf(str + 1, "%x", &c);
            return rgb8_packed(c);
        }
        else
        {
            named_color c;
            unsigned len = strlen(str);
            if(len > sizeof(c.name) - 1)
            {
                throw exception("parse_color: Invalid color name '%s'", str);
            }
            strcpy(c.name, str);
            const void* p = bsearch(&c, 
                                    colors, 
                                    sizeof(colors) / sizeof(colors[0]), 
                                    sizeof(colors[0]), 
                                    cmp_color);
            if(p == 0)
            {
                throw exception("parse_color: Invalid color name '%s'", str);
            }
            const named_color* pc = (const named_color*)p;
            return rgba8(pc->r, pc->g, pc->b, pc->a);
        }
    }

    double parse_double(const char* str)
    {
        while(*str == ' ') ++str;
        return atof(str);
    }



    //-------------------------------------------------------------
    bool parser::parse_attr(const char* name, const char* value)
    {
        if(strcmp(name, "style") == 0)
        {
            parse_style(value);
        }
        else
        if(strcmp(name, "fill") == 0)
        {
            if(strcmp(value, "none") == 0)
            {
                m_path.fill_none();
            }
            else
            {
                m_path.fill(parse_color(value));
            }
        }
        else
        if(strcmp(name, "fill-opacity") == 0)
        {
            m_path.fill_opacity(parse_double(value));
        }
        else
        if(strcmp(name, "stroke") == 0)
        {
            if(strcmp(value, "none") == 0)
            {
                m_path.stroke_none();
            }
            else
            {
                m_path.stroke(parse_color(value));
            }
        }
        else
        if(strcmp(name, "stroke-width") == 0)
        {
            m_path.stroke_width(parse_double(value));
        }
        else
        if(strcmp(name, "stroke-linecap") == 0)
        {
            if(strcmp(value, "butt") == 0)        m_path.line_cap(butt_cap);
            else if(strcmp(value, "round") == 0)  m_path.line_cap(round_cap);
            else if(strcmp(value, "square") == 0) m_path.line_cap(square_cap);
        }
        else
        if(strcmp(name, "stroke-linejoin") == 0)
        {
            if(strcmp(value, "miter") == 0)      m_path.line_join(miter_join);
            else if(strcmp(value, "round") == 0) m_path.line_join(round_join);
            else if(strcmp(value, "bevel") == 0) m_path.line_join(bevel_join);
        }
        else
        if(strcmp(name, "stroke-miterlimit") == 0)
        {
            m_path.miter_limit(parse_double(value));
        }
        else
        if(strcmp(name, "stroke-opacity") == 0)
        {
            m_path.stroke_opacity(parse_double(value));
        }
        else
        if(strcmp(name, "transform") == 0)
        {
            parse_transform(value);
        }
        //else
        //if(strcmp(el, "<OTHER_ATTRIBUTES>") == 0) 
        //{
        //}
        // . . .
        else
        {
            return false;
        }
        return true;
    }



    //-------------------------------------------------------------
    void parser::copy_name(const char* start, const char* end)
    {
        unsigned len = unsigned(end - start);
        if(m_attr_name_len == 0 || len > m_attr_name_len)
        {
            delete [] m_attr_name;
            m_attr_name = new char[len + 1];
            m_attr_name_len = len;
        }
        if(len) memcpy(m_attr_name, start, len);
        m_attr_name[len] = 0;
    }



    //-------------------------------------------------------------
    void parser::copy_value(const char* start, const char* end)
    {
        unsigned len = unsigned(end - start);
        if(m_attr_value_len == 0 || len > m_attr_value_len)
        {
            delete [] m_attr_value;
            m_attr_value = new char[len + 1];
            m_attr_value_len = len;
        }
        if(len) memcpy(m_attr_value, start, len);
        m_attr_value[len] = 0;
    }


    //-------------------------------------------------------------
    bool parser::parse_name_value(const char* nv_start, const char* nv_end)
    {
        const char* str = nv_start;
        while(str < nv_end && *str != ':') ++str;

        const char* val = str;

        // Right Trim
        while(str > nv_start && 
            (*str == ':' || isspace(*str))) --str;
        ++str;

        copy_name(nv_start, str);

        while(val < nv_end && (*val == ':' || isspace(*val))) ++val;
        
        copy_value(val, nv_end);
        return parse_attr(m_attr_name, m_attr_value);
    }



    //-------------------------------------------------------------
    void parser::parse_style(const char* str)
    {
        while(*str)
        {
            // Left Trim
            while(*str && isspace(*str)) ++str;
            const char* nv_start = str;
            while(*str && *str != ';') ++str;
            const char* nv_end = str;

            // Right Trim
            while(nv_end > nv_start && 
                (*nv_end == ';' || isspace(*nv_end))) --nv_end;
            ++nv_end;

            parse_name_value(nv_start, nv_end);
            if(*str) ++str;
        }

    }


    //-------------------------------------------------------------
    void parser::parse_rect(const char** attr)
    {
        int i;
        double x = 0.0;
        double y = 0.0;
        double w = 0.0;
        double h = 0.0;

        m_path.begin_path();
        for(i = 0; attr[i]; i += 2)
        {
            if(!parse_attr(attr[i], attr[i + 1]))
            {
                if(strcmp(attr[i], "x") == 0)      x = parse_double(attr[i + 1]);
                if(strcmp(attr[i], "y") == 0)      y = parse_double(attr[i + 1]);
                if(strcmp(attr[i], "width") == 0)  w = parse_double(attr[i + 1]);
                if(strcmp(attr[i], "height") == 0) h = parse_double(attr[i + 1]);
                // rx - to be implemented 
                // ry - to be implemented
            }
        }


        if(w != 0.0 && h != 0.0)
        {
            if(w < 0.0) throw exception("parse_rect: Invalid width: %f", w);
            if(h < 0.0) throw exception("parse_rect: Invalid height: %f", h);

            m_path.move_to(x,     y);
            m_path.line_to(x + w, y);
            m_path.line_to(x + w, y + h);
            m_path.line_to(x,     y + h);
            m_path.close_subpath();
        }
        m_path.end_path();
    }


    //-------------------------------------------------------------
    void parser::parse_line(const char** attr)
    {
        int i;
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;

        m_path.begin_path();
        for(i = 0; attr[i]; i += 2)
        {
            if(!parse_attr(attr[i], attr[i + 1]))
            {
                if(strcmp(attr[i], "x1") == 0) x1 = parse_double(attr[i + 1]);
                if(strcmp(attr[i], "y1") == 0) y1 = parse_double(attr[i + 1]);
                if(strcmp(attr[i], "x2") == 0) x2 = parse_double(attr[i + 1]);
                if(strcmp(attr[i], "y2") == 0) y2 = parse_double(attr[i + 1]);
            }
        }

        m_path.move_to(x1, y1);
        m_path.line_to(x2, y2);
        m_path.end_path();
    }


    //-------------------------------------------------------------
    void parser::parse_poly(const char** attr, bool close_flag)
    {
        int i;
        double x = 0.0;
        double y = 0.0;

        m_path.begin_path();
        for(i = 0; attr[i]; i += 2)
        {
            if(!parse_attr(attr[i], attr[i + 1]))
            {
                if(strcmp(attr[i], "points") == 0) 
                {
                    m_tokenizer.set_path_str(attr[i + 1]);
                    if(!m_tokenizer.next())
                    {
                        throw exception("parse_poly: Too few coordinates");
                    }
                    x = m_tokenizer.last_number();
                    if(!m_tokenizer.next())
                    {
                        throw exception("parse_poly: Too few coordinates");
                    }
                    y = m_tokenizer.last_number();
                    m_path.move_to(x, y);
                    while(m_tokenizer.next())
                    {
                        x = m_tokenizer.last_number();
                        if(!m_tokenizer.next())
                        {
                            throw exception("parse_poly: Odd number of coordinates");
                        }
                        y = m_tokenizer.last_number();
                        m_path.line_to(x, y);
                    }
                }
            }
        }
        m_path.end_path();
    }

    //-------------------------------------------------------------
    void parser::parse_transform(const char* str)
    {
        while(*str)
        {
            if(islower(*str))
            {
                if(strncmp(str, "matrix", 6) == 0)    str += parse_matrix(str);    else 
                if(strncmp(str, "translate", 9) == 0) str += parse_translate(str); else 
                if(strncmp(str, "rotate", 6) == 0)    str += parse_rotate(str);    else 
                if(strncmp(str, "scale", 5) == 0)     str += parse_scale(str);     else 
                if(strncmp(str, "skewX", 5) == 0)     str += parse_skew_x(str);    else 
                if(strncmp(str, "skewY", 5) == 0)     str += parse_skew_y(str);    else
                {
                    ++str;
                }
            }
            else
            {
                ++str;
            }
        }
    }


    //-------------------------------------------------------------
    static bool is_numeric(char c)
    {
        return strchr("0123456789+-.eE", c) != 0;
    }

    //-------------------------------------------------------------
    static unsigned parse_transform_args(const char* str, 
                                         double* args, 
                                         unsigned max_na, 
                                         unsigned* na)
    {
        *na = 0;
        const char* ptr = str;
        while(*ptr && *ptr != '(') ++ptr;
        if(*ptr == 0)
        {
            throw exception("parse_transform_args: Invalid syntax");
        }
        const char* end = ptr;
        while(*end && *end != ')') ++end;
        if(*end == 0)
        {
            throw exception("parse_transform_args: Invalid syntax");
        }

        while(ptr < end)
        {
            if(is_numeric(*ptr))
            {
                if(*na >= max_na)
                {
                    throw exception("parse_transform_args: Too many arguments");
                }
                args[(*na)++] = atof(ptr);
                while(ptr < end && is_numeric(*ptr)) ++ptr;
            }
            else
            {
                ++ptr;
            }
        }
        return unsigned(end - str);
    }

    //-------------------------------------------------------------
    unsigned parser::parse_matrix(const char* str)
    {
        double args[6];
        unsigned na = 0;
        unsigned len = parse_transform_args(str, args, 6, &na);
        if(na != 6)
        {
            throw exception("parse_matrix: Invalid number of arguments");
        }
        m_path.transform().premultiply(trans_affine(args[0], args[1], args[2], args[3], args[4], args[5]));
        return len;
    }

    //-------------------------------------------------------------
    unsigned parser::parse_translate(const char* str)
    {
        double args[2];
        unsigned na = 0;
        unsigned len = parse_transform_args(str, args, 2, &na);
        if(na == 1) args[1] = 0.0;
        m_path.transform().premultiply(trans_affine_translation(args[0], args[1]));
        return len;
    }

    //-------------------------------------------------------------
    unsigned parser::parse_rotate(const char* str)
    {
        double args[3];
        unsigned na = 0;
        unsigned len = parse_transform_args(str, args, 3, &na);
        if(na == 1) 
        {
            m_path.transform().premultiply(trans_affine_rotation(deg2rad(args[0])));
        }
        else if(na == 3)
        {
            trans_affine t = trans_affine_translation(-args[1], -args[2]);
            t *= trans_affine_rotation(deg2rad(args[0]));
            t *= trans_affine_translation(args[1], args[2]);
            m_path.transform().premultiply(t);
        }
        else
        {
            throw exception("parse_rotate: Invalid number of arguments");
        }
        return len;
    }

    //-------------------------------------------------------------
    unsigned parser::parse_scale(const char* str)
    {
        double args[2];
        unsigned na = 0;
        unsigned len = parse_transform_args(str, args, 2, &na);
        if(na == 1) args[1] = args[0];
        m_path.transform().premultiply(trans_affine_scaling(args[0], args[1]));
        return len;
    }

    //-------------------------------------------------------------
    unsigned parser::parse_skew_x(const char* str)
    {
        double arg;
        unsigned na = 0;
        unsigned len = parse_transform_args(str, &arg, 1, &na);
        m_path.transform().premultiply(trans_affine_skewing(deg2rad(arg), 0.0));
        return len;
    }

    //-------------------------------------------------------------
    unsigned parser::parse_skew_y(const char* str)
    {
        double arg;
        unsigned na = 0;
        unsigned len = parse_transform_args(str, &arg, 1, &na);
        m_path.transform().premultiply(trans_affine_skewing(0.0, deg2rad(arg)));
        return len;
    }

}
}


