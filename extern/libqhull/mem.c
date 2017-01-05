/*<html><pre>  -<a                             href="qh-mem.htm"
  >-------------------------------</a><a name="TOP">-</a>

  mem.c
    memory management routines for qhull

  This is a standalone program.

  To initialize memory:

    qh_meminit(stderr);
    qh_meminitbuffers(qh IStracing, qh_MEMalign, 7, qh_MEMbufsize,qh_MEMinitbuf);
    qh_memsize((int)sizeof(facetT));
    qh_memsize((int)sizeof(facetT));
    ...
    qh_memsetup();

  To free up all memory buffers:
    qh_memfreeshort(&curlong, &totlong);

  if qh_NOmem,
    malloc/free is used instead of mem.c

  notes:
    uses Quickfit algorithm (freelists for commonly allocated sizes)
    assumes small sizes for freelists (it discards the tail of memory buffers)

  see:
    qh-mem.htm and mem.h
    global.c (qh_initbuffers) for an example of using mem.c

  Copyright (c) 1993-2015 The Geometry Center.
  $Id: //main/2015/qhull/src/libqhull/mem.c#7 $$Change: 2065 $
  $DateTime: 2016/01/18 13:51:04 $$Author: bbarber $
*/

#include "user.h"  /* for QHULL_CRTDBG */
#include "mem.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef qhDEFlibqhull
typedef struct ridgeT ridgeT;
typedef struct facetT facetT;
#ifdef _MSC_VER  /* Microsoft Visual C++ -- warning level 4 */
#pragma warning( disable : 4127)  /* conditional expression is constant */
#pragma warning( disable : 4706)  /* assignment within conditional function */
#endif
void    qh_errexit(int exitcode, facetT *, ridgeT *);
void    qh_exit(int exitcode);
void    qh_fprintf(FILE *fp, int msgcode, const char *fmt, ... );
void    qh_fprintf_stderr(int msgcode, const char *fmt, ... );
void    qh_free(void *mem);
void   *qh_malloc(size_t size);
#endif

/*============ -global data structure ==============
    see mem.h for definition
*/

qhmemT qhmem= {0,0,0,0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0};     /* remove "= {0}" if this causes a compiler error */

#ifndef qh_NOmem

/*============= internal functions ==============*/

static int qh_intcompare(const void *i, const void *j);

/*========== functions in alphabetical order ======== */

/*-<a                             href="qh-mem.htm#TOC"
  >-------------------------------</a><a name="intcompare">-</a>

  qh_intcompare( i, j )
    used by qsort and bsearch to compare two integers
*/
static int qh_intcompare(const void *i, const void *j) {
  return(*((const int *)i) - *((const int *)j));
} /* intcompare */


/*-<a                             href="qh-mem.htm#TOC"
  >--------------------------------</a><a name="memalloc">-</a>

  qh_memalloc( insize )
    returns object of insize bytes
    qhmem is the global memory structure

  returns:
    pointer to allocated memory
    errors if insufficient memory

  notes:
    use explicit type conversion to avoid type warnings on some compilers
    actual object may be larger than insize
    use qh_memalloc_() for inline code for quick allocations
    logs allocations if 'T5'
    caller is responsible for freeing the memory.
    short memory is freed on shutdown by qh_memfreeshort unless qh_NOmem

  design:
    if size < qhmem.LASTsize
      if qhmem.freelists[size] non-empty
        return first object on freelist
      else
        round up request to size of qhmem.freelists[size]
        allocate new allocation buffer if necessary
        allocate object from allocation buffer
    else
      allocate object with qh_malloc() in user.c
*/
void *qh_memalloc(int insize) {
  void **freelistp, *newbuffer;
  int idx, size, n;
  int outsize, bufsize;
  void *object;

  if (insize<0) {
      qh_fprintf(qhmem.ferr, 6235, "qhull error (qh_memalloc): negative request size (%d).  Did int overflow due to high-D?\n", insize); /* WARN64 */
      qh_errexit(qhmem_ERRmem, NULL, NULL);
  }
  if (insize>=0 && insize <= qhmem.LASTsize) {
    idx= qhmem.indextable[insize];
    outsize= qhmem.sizetable[idx];
    qhmem.totshort += outsize;
    freelistp= qhmem.freelists+idx;
    if ((object= *freelistp)) {
      qhmem.cntquick++;
      qhmem.totfree -= outsize;
      *freelistp= *((void **)*freelistp);  /* replace freelist with next object */
#ifdef qh_TRACEshort
      n= qhmem.cntshort+qhmem.cntquick+qhmem.freeshort;
      if (qhmem.IStracing >= 5)
          qh_fprintf(qhmem.ferr, 8141, "qh_mem %p n %8d alloc quick: %d bytes (tot %d cnt %d)\n", object, n, outsize, qhmem.totshort, qhmem.cntshort+qhmem.cntquick-qhmem.freeshort);
#endif
      return(object);
    }else {
      qhmem.cntshort++;
      if (outsize > qhmem.freesize) {
        qhmem.totdropped += qhmem.freesize;
        if (!qhmem.curbuffer)
          bufsize= qhmem.BUFinit;
        else
          bufsize= qhmem.BUFsize;
        if (!(newbuffer= qh_malloc((size_t)bufsize))) {
          qh_fprintf(qhmem.ferr, 6080, "qhull error (qh_memalloc): insufficient memory to allocate short memory buffer (%d bytes)\n", bufsize);
          qh_errexit(qhmem_ERRmem, NULL, NULL);
        }
        *((void **)newbuffer)= qhmem.curbuffer;  /* prepend newbuffer to curbuffer
                                                    list.  newbuffer!=0 by QH6080 */
        qhmem.curbuffer= newbuffer;
        size= (sizeof(void **) + qhmem.ALIGNmask) & ~qhmem.ALIGNmask;
        qhmem.freemem= (void *)((char *)newbuffer+size);
        qhmem.freesize= bufsize - size;
        qhmem.totbuffer += bufsize - size; /* easier to check */
        /* Periodically test totbuffer.  It matches at beginning and exit of every call */
        n = qhmem.totshort + qhmem.totfree + qhmem.totdropped + qhmem.freesize - outsize;
        if (qhmem.totbuffer != n) {
            qh_fprintf(qhmem.ferr, 6212, "qh_memalloc internal error: short totbuffer %d != totshort+totfree... %d\n", qhmem.totbuffer, n);
            qh_errexit(qhmem_ERRmem, NULL, NULL);
        }
      }
      object= qhmem.freemem;
      qhmem.freemem= (void *)((char *)qhmem.freemem + outsize);
      qhmem.freesize -= outsize;
      qhmem.totunused += outsize - insize;
#ifdef qh_TRACEshort
      n= qhmem.cntshort+qhmem.cntquick+qhmem.freeshort;
      if (qhmem.IStracing >= 5)
          qh_fprintf(qhmem.ferr, 8140, "qh_mem %p n %8d alloc short: %d bytes (tot %d cnt %d)\n", object, n, outsize, qhmem.totshort, qhmem.cntshort+qhmem.cntquick-qhmem.freeshort);
#endif
      return object;
    }
  }else {                     /* long allocation */
    if (!qhmem.indextable) {
      qh_fprintf(qhmem.ferr, 6081, "qhull internal error (qh_memalloc): qhmem has not been initialized.\n");
      qh_errexit(qhmem_ERRqhull, NULL, NULL);
    }
    outsize= insize;
    qhmem.cntlong++;
    qhmem.totlong += outsize;
    if (qhmem.maxlong < qhmem.totlong)
      qhmem.maxlong= qhmem.totlong;
    if (!(object= qh_malloc((size_t)outsize))) {
      qh_fprintf(qhmem.ferr, 6082, "qhull error (qh_memalloc): insufficient memory to allocate %d bytes\n", outsize);
      qh_errexit(qhmem_ERRmem, NULL, NULL);
    }
    if (qhmem.IStracing >= 5)
      qh_fprintf(qhmem.ferr, 8057, "qh_mem %p n %8d alloc long: %d bytes (tot %d cnt %d)\n", object, qhmem.cntlong+qhmem.freelong, outsize, qhmem.totlong, qhmem.cntlong-qhmem.freelong);
  }
  return(object);
} /* memalloc */


/*-<a                             href="qh-mem.htm#TOC"
  >--------------------------------</a><a name="memcheck">-</a>

  qh_memcheck( )
*/
void qh_memcheck(void) {
  int i, count, totfree= 0;
  void *object;

  if (qhmem.ferr == 0 || qhmem.IStracing < 0 || qhmem.IStracing > 10 || (((qhmem.ALIGNmask+1) & qhmem.ALIGNmask) != 0)) {
    qh_fprintf_stderr(6244, "qh_memcheck error: either qhmem is overwritten or qhmem is not initialized.  Call qh_meminit() or qh_new_qhull() before calling qh_mem routines.  ferr 0x%x IsTracing %d ALIGNmask 0x%x", qhmem.ferr, qhmem.IStracing, qhmem.ALIGNmask);
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  if (qhmem.IStracing != 0)
    qh_fprintf(qhmem.ferr, 8143, "qh_memcheck: check size of freelists on qhmem\nqh_memcheck: A segmentation fault indicates an overwrite of qhmem\n");
  for (i=0; i < qhmem.TABLEsize; i++) {
    count=0;
    for (object= qhmem.freelists[i]; object; object= *((void **)object))
      count++;
    totfree += qhmem.sizetable[i] * count;
  }
  if (totfree != qhmem.totfree) {
    qh_fprintf(qhmem.ferr, 6211, "Qhull internal error (qh_memcheck): totfree %d not equal to freelist total %d\n", qhmem.totfree, totfree);
    qh_errexit(qhmem_ERRqhull, NULL, NULL);
  }
  if (qhmem.IStracing != 0)
    qh_fprintf(qhmem.ferr, 8144, "qh_memcheck: total size of freelists totfree is the same as qhmem.totfree\n", totfree);
} /* memcheck */

/*-<a                             href="qh-mem.htm#TOC"
  >--------------------------------</a><a name="memfree">-</a>

  qh_memfree( object, insize )
    free up an object of size bytes
    size is insize from qh_memalloc

  notes:
    object may be NULL
    type checking warns if using (void **)object
    use qh_memfree_() for quick free's of small objects

  design:
    if size <= qhmem.LASTsize
      append object to corresponding freelist
    else
      call qh_free(object)
*/
void qh_memfree(void *object, int insize) {
  void **freelistp;
  int idx, outsize;

  if (!object)
    return;
  if (insize <= qhmem.LASTsize) {
    qhmem.freeshort++;
    idx= qhmem.indextable[insize];
    outsize= qhmem.sizetable[idx];
    qhmem.totfree += outsize;
    qhmem.totshort -= outsize;
    freelistp= qhmem.freelists + idx;
    *((void **)object)= *freelistp;
    *freelistp= object;
#ifdef qh_TRACEshort
    idx= qhmem.cntshort+qhmem.cntquick+qhmem.freeshort;
    if (qhmem.IStracing >= 5)
        qh_fprintf(qhmem.ferr, 8142, "qh_mem %p n %8d free short: %d bytes (tot %d cnt %d)\n", object, idx, outsize, qhmem.totshort, qhmem.cntshort+qhmem.cntquick-qhmem.freeshort);
#endif
  }else {
    qhmem.freelong++;
    qhmem.totlong -= insize;
    if (qhmem.IStracing >= 5)
      qh_fprintf(qhmem.ferr, 8058, "qh_mem %p n %8d free long: %d bytes (tot %d cnt %d)\n", object, qhmem.cntlong+qhmem.freelong, insize, qhmem.totlong, qhmem.cntlong-qhmem.freelong);
    qh_free(object);
  }
} /* memfree */


/*-<a                             href="qh-mem.htm#TOC"
  >-------------------------------</a><a name="memfreeshort">-</a>

  qh_memfreeshort( curlong, totlong )
    frees up all short and qhmem memory allocations

  returns:
    number and size of current long allocations

  see:
    qh_freeqhull(allMem)
    qh_memtotal(curlong, totlong, curshort, totshort, maxlong, totbuffer);
*/
void qh_memfreeshort(int *curlong, int *totlong) {
  void *buffer, *nextbuffer;
  FILE *ferr;

  *curlong= qhmem.cntlong - qhmem.freelong;
  *totlong= qhmem.totlong;
  for (buffer= qhmem.curbuffer; buffer; buffer= nextbuffer) {
    nextbuffer= *((void **) buffer);
    qh_free(buffer);
  }
  qhmem.curbuffer= NULL;
  if (qhmem.LASTsize) {
    qh_free(qhmem.indextable);
    qh_free(qhmem.freelists);
    qh_free(qhmem.sizetable);
  }
  ferr= qhmem.ferr;
  memset((char *)&qhmem, 0, sizeof(qhmem));  /* every field is 0, FALSE, NULL */
  qhmem.ferr= ferr;
} /* memfreeshort */


/*-<a                             href="qh-mem.htm#TOC"
  >--------------------------------</a><a name="meminit">-</a>

  qh_meminit( ferr )
    initialize qhmem and test sizeof( void*)
    Does not throw errors.  qh_exit on failure
*/
void qh_meminit(FILE *ferr) {

  memset((char *)&qhmem, 0, sizeof(qhmem));  /* every field is 0, FALSE, NULL */
  if (ferr)
    qhmem.ferr= ferr;
  else
    qhmem.ferr= stderr;
  if (sizeof(void*) < sizeof(int)) {
    qh_fprintf(qhmem.ferr, 6083, "qhull internal error (qh_meminit): sizeof(void*) %d < sizeof(int) %d.  qset.c will not work\n", (int)sizeof(void*), (int)sizeof(int));
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  if (sizeof(void*) > sizeof(ptr_intT)) {
      qh_fprintf(qhmem.ferr, 6084, "qhull internal error (qh_meminit): sizeof(void*) %d > sizeof(ptr_intT) %d. Change ptr_intT in mem.h to 'long long'\n", (int)sizeof(void*), (int)sizeof(ptr_intT));
      qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  qh_memcheck();
} /* meminit */

/*-<a                             href="qh-mem.htm#TOC"
  >-------------------------------</a><a name="meminitbuffers">-</a>

  qh_meminitbuffers( tracelevel, alignment, numsizes, bufsize, bufinit )
    initialize qhmem
    if tracelevel >= 5, trace memory allocations
    alignment= desired address alignment for memory allocations
    numsizes= number of freelists
    bufsize=  size of additional memory buffers for short allocations
    bufinit=  size of initial memory buffer for short allocations
*/
void qh_meminitbuffers(int tracelevel, int alignment, int numsizes, int bufsize, int bufinit) {

  qhmem.IStracing= tracelevel;
  qhmem.NUMsizes= numsizes;
  qhmem.BUFsize= bufsize;
  qhmem.BUFinit= bufinit;
  qhmem.ALIGNmask= alignment-1;
  if (qhmem.ALIGNmask & ~qhmem.ALIGNmask) {
    qh_fprintf(qhmem.ferr, 6085, "qhull internal error (qh_meminit): memory alignment %d is not a power of 2\n", alignment);
    qh_errexit(qhmem_ERRqhull, NULL, NULL);
  }
  qhmem.sizetable= (int *) calloc((size_t)numsizes, sizeof(int));
  qhmem.freelists= (void **) calloc((size_t)numsizes, sizeof(void *));
  if (!qhmem.sizetable || !qhmem.freelists) {
    qh_fprintf(qhmem.ferr, 6086, "qhull error (qh_meminit): insufficient memory\n");
    qh_errexit(qhmem_ERRmem, NULL, NULL);
  }
  if (qhmem.IStracing >= 1)
    qh_fprintf(qhmem.ferr, 8059, "qh_meminitbuffers: memory initialized with alignment %d\n", alignment);
} /* meminitbuffers */

/*-<a                             href="qh-mem.htm#TOC"
  >-------------------------------</a><a name="memsetup">-</a>

  qh_memsetup()
    set up memory after running memsize()
*/
void qh_memsetup(void) {
  int k,i;

  qsort(qhmem.sizetable, (size_t)qhmem.TABLEsize, sizeof(int), qh_intcompare);
  qhmem.LASTsize= qhmem.sizetable[qhmem.TABLEsize-1];
  if (qhmem.LASTsize >= qhmem.BUFsize || qhmem.LASTsize >= qhmem.BUFinit) {
    qh_fprintf(qhmem.ferr, 6087, "qhull error (qh_memsetup): largest mem size %d is >= buffer size %d or initial buffer size %d\n",
            qhmem.LASTsize, qhmem.BUFsize, qhmem.BUFinit);
    qh_errexit(qhmem_ERRmem, NULL, NULL);
  }
  if (!(qhmem.indextable= (int *)qh_malloc((qhmem.LASTsize+1) * sizeof(int)))) {
    qh_fprintf(qhmem.ferr, 6088, "qhull error (qh_memsetup): insufficient memory\n");
    qh_errexit(qhmem_ERRmem, NULL, NULL);
  }
  for (k=qhmem.LASTsize+1; k--; )
    qhmem.indextable[k]= k;
  i= 0;
  for (k=0; k <= qhmem.LASTsize; k++) {
    if (qhmem.indextable[k] <= qhmem.sizetable[i])
      qhmem.indextable[k]= i;
    else
      qhmem.indextable[k]= ++i;
  }
} /* memsetup */

/*-<a                             href="qh-mem.htm#TOC"
  >-------------------------------</a><a name="memsize">-</a>

  qh_memsize( size )
    define a free list for this size
*/
void qh_memsize(int size) {
  int k;

  if (qhmem.LASTsize) {
    qh_fprintf(qhmem.ferr, 6089, "qhull error (qh_memsize): called after qhmem_setup\n");
    qh_errexit(qhmem_ERRqhull, NULL, NULL);
  }
  size= (size + qhmem.ALIGNmask) & ~qhmem.ALIGNmask;
  for (k=qhmem.TABLEsize; k--; ) {
    if (qhmem.sizetable[k] == size)
      return;
  }
  if (qhmem.TABLEsize < qhmem.NUMsizes)
    qhmem.sizetable[qhmem.TABLEsize++]= size;
  else
    qh_fprintf(qhmem.ferr, 7060, "qhull warning (memsize): free list table has room for only %d sizes\n", qhmem.NUMsizes);
} /* memsize */


/*-<a                             href="qh-mem.htm#TOC"
  >-------------------------------</a><a name="memstatistics">-</a>

  qh_memstatistics( fp )
    print out memory statistics

    Verifies that qhmem.totfree == sum of freelists
*/
void qh_memstatistics(FILE *fp) {
  int i;
  int count;
  void *object;

  qh_memcheck();
  qh_fprintf(fp, 9278, "\nmemory statistics:\n\
%7d quick allocations\n\
%7d short allocations\n\
%7d long allocations\n\
%7d short frees\n\
%7d long frees\n\
%7d bytes of short memory in use\n\
%7d bytes of short memory in freelists\n\
%7d bytes of dropped short memory\n\
%7d bytes of unused short memory (estimated)\n\
%7d bytes of long memory allocated (max, except for input)\n\
%7d bytes of long memory in use (in %d pieces)\n\
%7d bytes of short memory buffers (minus links)\n\
%7d bytes per short memory buffer (initially %d bytes)\n",
           qhmem.cntquick, qhmem.cntshort, qhmem.cntlong,
           qhmem.freeshort, qhmem.freelong,
           qhmem.totshort, qhmem.totfree,
           qhmem.totdropped + qhmem.freesize, qhmem.totunused,
           qhmem.maxlong, qhmem.totlong, qhmem.cntlong - qhmem.freelong,
           qhmem.totbuffer, qhmem.BUFsize, qhmem.BUFinit);
  if (qhmem.cntlarger) {
    qh_fprintf(fp, 9279, "%7d calls to qh_setlarger\n%7.2g     average copy size\n",
           qhmem.cntlarger, ((float)qhmem.totlarger)/(float)qhmem.cntlarger);
    qh_fprintf(fp, 9280, "  freelists(bytes->count):");
  }
  for (i=0; i < qhmem.TABLEsize; i++) {
    count=0;
    for (object= qhmem.freelists[i]; object; object= *((void **)object))
      count++;
    qh_fprintf(fp, 9281, " %d->%d", qhmem.sizetable[i], count);
  }
  qh_fprintf(fp, 9282, "\n\n");
} /* memstatistics */


/*-<a                             href="qh-mem.htm#TOC"
  >-------------------------------</a><a name="NOmem">-</a>

  qh_NOmem
    turn off quick-fit memory allocation

  notes:
    uses qh_malloc() and qh_free() instead
*/
#else /* qh_NOmem */

void *qh_memalloc(int insize) {
  void *object;

  if (!(object= qh_malloc((size_t)insize))) {
    qh_fprintf(qhmem.ferr, 6090, "qhull error (qh_memalloc): insufficient memory\n");
    qh_errexit(qhmem_ERRmem, NULL, NULL);
  }
  qhmem.cntlong++;
  qhmem.totlong += insize;
  if (qhmem.maxlong < qhmem.totlong)
      qhmem.maxlong= qhmem.totlong;
  if (qhmem.IStracing >= 5)
    qh_fprintf(qhmem.ferr, 8060, "qh_mem %p n %8d alloc long: %d bytes (tot %d cnt %d)\n", object, qhmem.cntlong+qhmem.freelong, insize, qhmem.totlong, qhmem.cntlong-qhmem.freelong);
  return object;
}

void qh_memfree(void *object, int insize) {

  if (!object)
    return;
  qh_free(object);
  qhmem.freelong++;
  qhmem.totlong -= insize;
  if (qhmem.IStracing >= 5)
    qh_fprintf(qhmem.ferr, 8061, "qh_mem %p n %8d free long: %d bytes (tot %d cnt %d)\n", object, qhmem.cntlong+qhmem.freelong, insize, qhmem.totlong, qhmem.cntlong-qhmem.freelong);
}

void qh_memfreeshort(int *curlong, int *totlong) {
  *totlong= qhmem.totlong;
  *curlong= qhmem.cntlong - qhmem.freelong;
  memset((char *)&qhmem, 0, sizeof(qhmem));  /* every field is 0, FALSE, NULL */
}

void qh_meminit(FILE *ferr) {

  memset((char *)&qhmem, 0, sizeof(qhmem));  /* every field is 0, FALSE, NULL */
  if (ferr)
      qhmem.ferr= ferr;
  else
      qhmem.ferr= stderr;
  if (sizeof(void*) < sizeof(int)) {
    qh_fprintf(qhmem.ferr, 6091, "qhull internal error (qh_meminit): sizeof(void*) %d < sizeof(int) %d.  qset.c will not work\n", (int)sizeof(void*), (int)sizeof(int));
    qh_errexit(qhmem_ERRqhull, NULL, NULL);
  }
}

void qh_meminitbuffers(int tracelevel, int alignment, int numsizes, int bufsize, int bufinit) {

  qhmem.IStracing= tracelevel;
}

void qh_memsetup(void) {

}

void qh_memsize(int size) {

}

void qh_memstatistics(FILE *fp) {

  qh_fprintf(fp, 9409, "\nmemory statistics:\n\
%7d long allocations\n\
%7d long frees\n\
%7d bytes of long memory allocated (max, except for input)\n\
%7d bytes of long memory in use (in %d pieces)\n",
           qhmem.cntlong,
           qhmem.freelong,
           qhmem.maxlong, qhmem.totlong, qhmem.cntlong - qhmem.freelong);
}

#endif /* qh_NOmem */

/*-<a                             href="qh-mem.htm#TOC"
>-------------------------------</a><a name="memtotlong">-</a>

  qh_memtotal( totlong, curlong, totshort, curshort, maxlong, totbuffer )
    Return the total, allocated long and short memory

  returns:
    Returns the total current bytes of long and short allocations
    Returns the current count of long and short allocations
    Returns the maximum long memory and total short buffer (minus one link per buffer)
    Does not error (UsingLibQhull.cpp)
*/
void qh_memtotal(int *totlong, int *curlong, int *totshort, int *curshort, int *maxlong, int *totbuffer) {
    *totlong= qhmem.totlong;
    *curlong= qhmem.cntlong - qhmem.freelong;
    *totshort= qhmem.totshort;
    *curshort= qhmem.cntshort + qhmem.cntquick - qhmem.freeshort;
    *maxlong= qhmem.maxlong;
    *totbuffer= qhmem.totbuffer;
} /* memtotlong */

