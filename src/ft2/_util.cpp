#include "_util.h"

namespace matplotlib::ft2 {

namespace detail {
// Load FreeType error codes.  This approach (modified to use
// std::unordered_map) is documented in fterror.h.
// NOTE: If we require FreeType>=2.6.3 then the macro can be replaced by
// FTERRORS_H_.
#undef __FTERRORS_H__
#define FT_ERRORDEF( e, v, s )  { e, s },
#define FT_ERROR_START_LIST     {
#define FT_ERROR_END_LIST       };

std::unordered_map<FT_Error, std::string> ft_errors =

#include FT_ERRORS_H
}

}
