/*<html><pre>  -<a                             href="qh-user_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   userprintf_rbox_r.c
   user redefinable function -- qh_fprintf_rbox

   see README.txt  see COPYING.txt for copyright information.

   If you recompile and load this file, then userprintf_rbox_r.o will not be loaded
   from qhull.a or qhull.lib

   See libqhull_r.h for data structures, macros, and user-callable functions.
   See user_r.c for qhull-related, redefinable functions
   see user_r.h for user-definable constants
   See usermem_r.c for qh_exit(), qh_free(), and qh_malloc()
   see Qhull.cpp and RboxPoints.cpp for examples.

   Please report any errors that you fix to qhull@qhull.org
*/

#include "libqhull_r.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/*-<a                             href="qh-user_r.htm#TOC"
   >-------------------------------</a><a name="qh_fprintf_rbox">-</a>

   qh_fprintf_rbox(qh, fp, msgcode, format, list of args )
     print arguments to *fp according to format
     Use qh_fprintf_rbox() for rboxlib_r.c

   notes:
     same as fprintf()
     fgets() is not trapped like fprintf()
     exit qh_fprintf_rbox via qh_errexit_rbox()
*/

void qh_fprintf_rbox(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... ) {
    va_list args;

    if (!fp) {
      qh_fprintf_stderr(6231, "qhull internal error (userprintf_rbox_r.c): fp is 0.  Wrong qh_fprintf_rbox called.\n");
      qh_errexit_rbox(qh, qh_ERRqhull);
    }
    if (msgcode >= MSG_ERROR && msgcode < MSG_STDERR)
      fprintf(fp, "QH%.4d ", msgcode);
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
} /* qh_fprintf_rbox */

