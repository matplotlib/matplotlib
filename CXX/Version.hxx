//
//  CXX/Version.hxx
//
#ifndef __PyCXX_version_hxx__
#define __PyCXX_version_hxx__

#define PYCXX_VERSION_MAJOR 5
#define PYCXX_VERSION_MINOR 3
#define PYCXX_VERSION_PATCH 3
#define PYCXX_MAKEVERSION( major, minor, patch ) ((major<<16)|(minor<<8)|(patch))
#define PYCXX_VERSION PYCXX_MAKEVERSION( PYCXX_VERSION_MAJOR, PYCXX_VERSION_MINOR, PYCXX_VERSION_PATCH )
#endif
