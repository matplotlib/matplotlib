#ifndef __PyCXX_config_hh__
#define __PyCXX_config_hh__

//
// Microsoft VC++ 6.0 has no traits
//
#if defined( _MSC_VER )

#  define STANDARD_LIBRARY_HAS_ITERATOR_TRAITS 1

#elif defined( __GNUC__ )
#  if __GNUC__ >= 3
#    define STANDARD_LIBRARY_HAS_ITERATOR_TRAITS 1
#  else
#    define STANDARD_LIBRARY_HAS_ITERATOR_TRAITS 0
#endif

//
//	Assume all other compilers do
//
#else

// Macros to deal with deficiencies in compilers
#  define STANDARD_LIBRARY_HAS_ITERATOR_TRAITS 1
#endif

#if STANDARD_LIBRARY_HAS_ITERATOR_TRAITS
#  define random_access_iterator_parent(itemtype) std::iterator<std::random_access_iterator_tag,itemtype,int>
#else
#  define random_access_iterator_parent(itemtype) std::random_access_iterator<itemtype, int>
#endif

//
//	Which C++ standard is in use?
//
#if defined( _MSC_VER )
#  if _MSC_VER <= 1200
// MSVC++ 6.0
#    define PYCXX_ISO_CPP_LIB 0
#    define STR_STREAM <strstream>
#    define TEMPLATE_TYPENAME class
#  else
#    define PYCXX_ISO_CPP_LIB 1
#    define STR_STREAM <sstream>
#    define TEMPLATE_TYPENAME typename
#  endif
#elif defined( __GNUC__ )
#  if __GNUC__ >= 3
#    define PYCXX_ISO_CPP_LIB 1
#    define STR_STREAM <sstream>
#    define TEMPLATE_TYPENAME typename
#  else
#    define PYCXX_ISO_CPP_LIB 0
#    define STR_STREAM <strstream>
#    define TEMPLATE_TYPENAME class
#  endif
#endif

#if PYCXX_ISO_CPP_LIB
#    define STR_STREAM <sstream>
#    define OSTRSTREAM ostringstream
#    define EXPLICIT_TYPENAME typename
#    define EXPLICIT_CLASS class
#    define TEMPLATE_TYPENAME typename
#else
#    define STR_STREAM <strstream>
#    define OSTRSTREAM ostrstream
#    define EXPLICIT_TYPENAME
#    define EXPLICIT_CLASS
#    define TEMPLATE_TYPENAME class
#endif


#endif //  __PyCXX_config_hh__
