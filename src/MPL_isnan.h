/*

These definitions were inspired by and originally modified from
numarray's Include/numarray/nummacro.h

The "64" below refers to double precision floating point numbers. This
code only works on doubles.

*/

#if defined(SIZEOF_VOID_P)
#if SIZEOF_VOID_P == 8
#define MPL_LP64 1
#else
#define MPL_LP64 0
#endif
#else
#define MPL_LP64 0
#endif

#if MPL_LP64
typedef long int                 MPL_Int64;
#else                  /* 32-bit platforms */
#if defined(_MSC_VER)
typedef __int64                  MPL_Int64;
#else
typedef long long                MPL_Int64;
#endif
#endif

#if !defined(MPL_U64)
#define MPL_U64(u) (* (MPL_Int64 *) &(u) )
#endif /* MPL_U64 */

#if !defined(MPL_isnan64)
#if !defined(_MSC_VER)
#define MPL_isnan64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)  && ((MPL_U64(u) &  0x000fffffffffffffLL) != 0)) ? 1:0
#else
#define MPL_isnan64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((MPL_U64(u) & 0x000fffffffffffffi64) != 0)) ? 1:0
#endif
#endif /* MPL_isnan64 */

#if !defined(MPL_isinf64)
#if !defined(_MSC_VER)
#define MPL_isinf64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)  && ((MPL_U64(u) &  0x000fffffffffffffLL) == 0)) ? 1:0
#else
#define MPL_isinf64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((MPL_U64(u) & 0x000fffffffffffffi64) == 0)) ? 1:0
#endif
#endif /* MPL_isinf64 */

#if !defined(MPL_isfinite64)
#if !defined(_MSC_VER)
#define MPL_isfinite64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL)  != 0x7ff0000000000000LL)) ? 1:0
#else
#define MPL_isfinite64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) != 0x7ff0000000000000i64)) ? 1:0
#endif
#endif /* MPL_isfinite64 */
