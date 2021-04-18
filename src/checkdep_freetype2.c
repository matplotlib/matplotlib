#ifdef __has_include
  #if !__has_include(<ft2build.h>)
    #error "FreeType version 2.3 or higher is required. \
You may unset the system_freetype entry in setup.cfg to let Matplotlib download it."
  #endif
#endif

#include <ft2build.h>
#include FT_FREETYPE_H

#define XSTR(x) STR(x)
#define STR(x) #x

#pragma message("Compiling with FreeType version " \
  XSTR(FREETYPE_MAJOR) "." XSTR(FREETYPE_MINOR) "." XSTR(FREETYPE_PATCH) ".")
#if FREETYPE_MAJOR << 16 + FREETYPE_MINOR << 8 + FREETYPE_PATCH < 0x020300
  #error "FreeType version 2.3 or higher is required. \
You may unset the system_freetype entry in setup.cfg to let Matplotlib download it."
#endif
