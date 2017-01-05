/*<html><pre>  -<a                             href="index.htm#TOC"
  >-------------------------------</a><a name="TOP">-</a>

   random.c and utilities
     Park & Miller's minimimal standard random number generator
     argc/argv conversion

     Used by rbox.  Do not use 'qh' 
*/

#include "libqhull.h"
#include "random.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER  /* Microsoft Visual C++ -- warning level 4 */
#pragma warning( disable : 4706)  /* assignment within conditional function */
#pragma warning( disable : 4996)  /* function was declared deprecated(strcpy, localtime, etc.) */
#endif

/*-<a                             href="qh-globa.htm#TOC"
 >-------------------------------</a><a name="argv_to_command">-</a>

 qh_argv_to_command( argc, argv, command, max_size )

    build command from argc/argv
    max_size is at least

 returns:
    a space-delimited string of options (just as typed)
    returns false if max_size is too short

 notes:
    silently removes
    makes option string easy to input and output
    matches qh_argv_to_command_size()

    argc may be 0
*/
int qh_argv_to_command(int argc, char *argv[], char* command, int max_size) {
  int i, remaining;
  char *s;
  *command= '\0';  /* max_size > 0 */

  if (argc) {
    if ((s= strrchr( argv[0], '\\')) /* get filename w/o .exe extension */
    || (s= strrchr( argv[0], '/')))
        s++;
    else
        s= argv[0];
    if ((int)strlen(s) < max_size)   /* WARN64 */
        strcpy(command, s);
    else
        goto error_argv;
    if ((s= strstr(command, ".EXE"))
    ||  (s= strstr(command, ".exe")))
        *s= '\0';
  }
  for (i=1; i < argc; i++) {
    s= argv[i];
    remaining= max_size - (int)strlen(command) - (int)strlen(s) - 2;   /* WARN64 */
    if (!*s || strchr(s, ' ')) {
      char *t= command + strlen(command);
      remaining -= 2;
      if (remaining < 0) {
        goto error_argv;
      }
      *t++= ' ';
      *t++= '"';
      while (*s) {
        if (*s == '"') {
          if (--remaining < 0)
            goto error_argv;
          *t++= '\\';
        }
        *t++= *s++;
      }
      *t++= '"';
      *t= '\0';
    }else if (remaining < 0) {
      goto error_argv;
    }else
      strcat(command, " ");
      strcat(command, s);
  }
  return 1;

error_argv:
  return 0;
} /* argv_to_command */

/*-<a                             href="qh-globa.htm#TOC"
>-------------------------------</a><a name="argv_to_command_size">-</a>

qh_argv_to_command_size( argc, argv )

    return size to allocate for qh_argv_to_command()

notes:
    argc may be 0
    actual size is usually shorter
*/
int qh_argv_to_command_size(int argc, char *argv[]) {
    unsigned int count= 1; /* null-terminator if argc==0 */
    int i;
    char *s;

    for (i=0; i<argc; i++){
      count += (int)strlen(argv[i]) + 1;   /* WARN64 */
      if (i>0 && strchr(argv[i], ' ')) {
        count += 2;  /* quote delimiters */
        for (s=argv[i]; *s; s++) {
          if (*s == '"') {
            count++;
          }
        }
      }
    }
    return count;
} /* argv_to_command_size */

/*-<a                             href="qh-geom.htm#TOC"
  >-------------------------------</a><a name="rand">-</a>

  qh_rand()
  qh_srand( seed )
    generate pseudo-random number between 1 and 2^31 -2

  notes:
    For qhull and rbox, called from qh_RANDOMint(),etc. [user.h]

    From Park & Miller's minimal standard random number generator
      Communications of the ACM, 31:1192-1201, 1988.
    Does not use 0 or 2^31 -1
      this is silently enforced by qh_srand()
    Can make 'Rn' much faster by moving qh_rand to qh_distplane
*/

/* Global variables and constants */

int qh_last_random= 1;  /* define as global variable instead of using qh */

#define qh_rand_a 16807
#define qh_rand_m 2147483647
#define qh_rand_q 127773  /* m div a */
#define qh_rand_r 2836    /* m mod a */

int qh_rand( void) {
    int lo, hi, test;
    int seed = qh_last_random;

    hi = seed / qh_rand_q;  /* seed div q */
    lo = seed % qh_rand_q;  /* seed mod q */
    test = qh_rand_a * lo - qh_rand_r * hi;
    if (test > 0)
        seed= test;
    else
        seed= test + qh_rand_m;
    qh_last_random= seed;
    /* seed = seed < qh_RANDOMmax/2 ? 0 : qh_RANDOMmax;  for testing */
    /* seed = qh_RANDOMmax;  for testing */
    return seed;
} /* rand */

void qh_srand( int seed) {
    if (seed < 1)
        qh_last_random= 1;
    else if (seed >= qh_rand_m)
        qh_last_random= qh_rand_m - 1;
    else
        qh_last_random= seed;
} /* qh_srand */

/*-<a                             href="qh-geom.htm#TOC"
>-------------------------------</a><a name="randomfactor">-</a>

qh_randomfactor( scale, offset )
  return a random factor r * scale + offset

notes:
  qh.RANDOMa/b are defined in global.c
*/
realT qh_randomfactor(realT scale, realT offset) {
    realT randr;

    randr= qh_RANDOMint;
    return randr * scale + offset;
} /* randomfactor */

/*-<a                             href="qh-geom.htm#TOC"
>-------------------------------</a><a name="randommatrix">-</a>

qh_randommatrix( buffer, dim, rows )
  generate a random dim X dim matrix in range [-1,1]
  assumes buffer is [dim+1, dim]

returns:
  sets buffer to random numbers
  sets rows to rows of buffer
  sets row[dim] as scratch row
*/
void qh_randommatrix(realT *buffer, int dim, realT **rows) {
    int i, k;
    realT **rowi, *coord, realr;

    coord= buffer;
    rowi= rows;
    for (i=0; i < dim; i++) {
        *(rowi++)= coord;
        for (k=0; k < dim; k++) {
            realr= qh_RANDOMint;
            *(coord++)= 2.0 * realr/(qh_RANDOMmax+1) - 1.0;
        }
    }
    *rowi= coord;
} /* randommatrix */

/*-<a                             href="qh-globa.htm#TOC"
  >-------------------------------</a><a name="strtol">-</a>

  qh_strtol( s, endp) qh_strtod( s, endp)
    internal versions of strtol() and strtod()
    does not skip trailing spaces
  notes:
    some implementations of strtol()/strtod() skip trailing spaces
*/
double qh_strtod(const char *s, char **endp) {
  double result;

  result= strtod(s, endp);
  if (s < (*endp) && (*endp)[-1] == ' ')
    (*endp)--;
  return result;
} /* strtod */

int qh_strtol(const char *s, char **endp) {
  int result;

  result= (int) strtol(s, endp, 10);     /* WARN64 */
  if (s< (*endp) && (*endp)[-1] == ' ')
    (*endp)--;
  return result;
} /* strtol */
