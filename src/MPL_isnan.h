// These definitions modified from numarray's Include/numarray/nummacro.h

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

#if !defined(MPL_isnan64)
#define MPL_U64(u) (* (MPL_Int64 *) &(u) )

#if !defined(_MSC_VER)
#define MPL_isnan64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL) == 0x7ff0000000000000LL)  && ((MPL_U64(u) & 0x000fffffffffffffLL) != 0)) ? 1:0
#else
#define MPL_isnan64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((MPL_U64(u) & 0x000fffffffffffffi64) != 0)) ? 1:0
#endif
#endif /* MPL_isnan64 */
