/* -*- mode: c++; c-basic-offset: 4 -*- */

#include <iostream>
#include <cstdarg>
#include <cstdio>
#include "mplutils.h"

void _VERBOSE(const std::string& s)
{
#ifdef VERBOSE
    std::cout << s << std::endl;
#endif
}


Printf::Printf(const char *fmt, ...)
    : buffer(new char[1024]) // some reasonably large number
{
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buffer, fmt, ap);
    va_end(ap);  // look ma - I rememberd it this time
}

Printf::~Printf()
{
    delete [] buffer;
}


std::ostream &operator<<(std::ostream &o, const Printf &p)
{
    return o << p.buffer;
}
