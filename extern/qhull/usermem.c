/*<html><pre>  -<a                             href="qh-user.htm"
  >-------------------------------</a><a name="TOP">-</a>

   usermem.c
   qh_exit(), qh_free(), and qh_malloc()

   See README.txt.

   If you redefine one of these functions you must redefine all of them.
   If you recompile and load this file, then usermem.o will not be loaded
   from qhull.a or qhull.lib

   See libqhull.h for data structures, macros, and user-callable functions.
   See user.c for qhull-related, redefinable functions
   see user.h for user-definable constants
   See userprintf.c for qh_fprintf and userprintf_rbox,c for qh_fprintf_rbox

   Please report any errors that you fix to qhull@qhull.org
*/

#include "libqhull.h"

#include <stdlib.h>

/*-<a                             href="qh-user.htm#TOC"
  >-------------------------------</a><a name="qh_exit">-</a>

  qh_exit( exitcode )
    exit program

  notes:
    same as exit()
*/
void qh_exit(int exitcode) {
    exit(exitcode);
} /* exit */

/*-<a                             href="qh-user.htm#TOC"
>-------------------------------</a><a name="qh_free">-</a>

qh_free( mem )
free memory

notes:
same as free()
*/
void qh_free(void *mem) {
    free(mem);
} /* free */

/*-<a                             href="qh-user.htm#TOC"
    >-------------------------------</a><a name="qh_malloc">-</a>

    qh_malloc( mem )
      allocate memory

    notes:
      same as malloc()
*/
void *qh_malloc(size_t size) {
    return malloc(size);
} /* malloc */
