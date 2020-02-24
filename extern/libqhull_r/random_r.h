/*<html><pre>  -<a                             href="qh-geom_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

  random_r.h
    header file for random and utility routines

   see qh-geom_r.htm and random_r.c

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/random_r.h#2 $$Change: 2666 $
   $DateTime: 2019/05/30 10:11:25 $$Author: bbarber $
*/

#ifndef qhDEFrandom
#define qhDEFrandom 1

#include "libqhull_r.h"

/*============= prototypes in alphabetical order ======= */

#ifdef __cplusplus
extern "C" {
#endif

int     qh_argv_to_command(int argc, char *argv[], char* command, int max_size);
int     qh_argv_to_command_size(int argc, char *argv[]);
int     qh_rand(qhT *qh);
void    qh_srand(qhT *qh, int seed);
realT   qh_randomfactor(qhT *qh, realT scale, realT offset);
void    qh_randommatrix(qhT *qh, realT *buffer, int dim, realT **row);
int     qh_strtol(const char *s, char **endp);
double  qh_strtod(const char *s, char **endp);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* qhDEFrandom */



