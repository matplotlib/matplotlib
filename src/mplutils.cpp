#include <iostream>
#include "mplutils.h"


void _VERBOSE(const std::string& s) {
#ifdef VERBOSE
  std::cout << s << std::endl;
#endif
}
