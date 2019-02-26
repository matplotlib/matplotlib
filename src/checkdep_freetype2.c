#include <ft2build.h>
#include FT_FREETYPE_H

#define XSTR(x) STR(x)
#define STR(x) #x

#pragma message("Compiling with FreeType version " \
  XSTR(FREETYPE_MAJOR) "." XSTR(FREETYPE_MINOR) "." XSTR(FREETYPE_PATCH) ".")
#if FREETYPE_MAJOR << 16 + FREETYPE_MINOR << 8 + FREETYPE_PATCH < 0x020300
    #error "FreeType version 2.3 or higher is required." \
      "Consider setting the MPLLOCALFREETYPE environment variable to 1."
  #error
#endif
