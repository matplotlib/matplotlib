#ifdef __has_include
  #if !__has_include(<png.h>)
    #error "libpng version 1.2 or higher is required. \
Consider installing it with e.g. 'conda install libpng', \
'apt install libpng12-dev', 'dnf install libpng-devel', or \
'brew install libpng'."
  #endif
#endif

#include <png.h>
#pragma message("Compiling with libpng version " PNG_LIBPNG_VER_STRING ".")
#if PNG_LIBPNG_VER < 10200
  #error "libpng version 1.2 or higher is required. \
Consider installing it with e.g. 'conda install libpng', \
'apt install libpng12-dev', 'dnf install libpng-devel', or \
'brew install libpng'."
#endif
