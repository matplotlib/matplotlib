/*<html><pre>  -<a                             href="qh-mem_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

  mem_r.c
    memory management routines for qhull

  See libqhull/mem.c for a standalone program.

  To initialize memory:

    qh_meminit(qh, stderr);
    qh_meminitbuffers(qh, qh->IStracing, qh_MEMalign, 7, qh_MEMbufsize,qh_MEMinitbuf);
    qh_memsize(qh, (int)sizeof(facetT));
    qh_memsize(qh, (int)sizeof(facetT));
    ...
    qh_memsetup(qh);

  To free up all memory buffers:
    qh_memfreeshort(qh, &curlong, &totlong);

  if qh_NOmem,
    malloc/free is used instead of mem_r.c

  notes:
    uses Quickfit algorithm (freelists for commonly allocated sizes)
    assumes small sizes for freelists (it discards the tail of memory buffers)

  see:
    qh-mem_r.htm and mem_r.h
    global_r.c (qh_initbuffers) for an example of using mem_r.c

  Copyright (c) 1993-2019 The Geometry Center.
  $Id: //main/2019/qhull/src/libqhull_r/mem_r.c#6 $$Change: 2711 $
  $DateTime: 2019/06/27 22:34:56 $$Author: bbarber $
*/

#include "libqhull_r.h"  /* includes user_r.h and mem_r.h */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef qh_NOmem

/*============= internal functions ==============*/

static int qh_intcompare(const void *i, const void *j);

/*========== functions in alphabetical order ======== */

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="intcompare">-</a>

  qh_intcompare( i, j )
    used by qsort and bsearch to compare two integers
*/
static int qh_intcompare(const void *i, const void *j) {
  return(*((const int *)i) - *((const int *)j));
} /* intcompare */


/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memalloc">-</a>

  qh_memalloc(qh, insize )
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
    if size < qh->qhmem.LASTsize
      if qh->qhmem.freelists[size] non-empty
        return first object on freelist
      else
        round up request to size of qh->qhmem.freelists[size]
        allocate new allocation buffer if necessary
        allocate object from allocation buffer
    else
      allocate object with qh_malloc() in user_r.c
*/
void *qh_memalloc(qhT *qh, int insize) {
  void **freelistp, *newbuffer;
  int idx, size, n;
  int outsize, bufsize;
  void *object;

  if (insize<0) {
      qh_fprintf(qh, qh->qhmem.ferr, 6235, "qhull error (qh_memalloc): negative request size (%d).  Did int overflow due to high-D?\n", insize); /* WARN64 */
      qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
  }
  if (insize>=0 && insize <= qh->qhmem.LASTsize) {
    idx= qh->qhmem.indextable[insize];
    outsize= qh->qhmem.sizetable[idx];
    qh->qhmem.totshort += outsize;
    freelistp= qh->qhmem.freelists+idx;
    if ((object= *freelistp)) {
      qh->qhmem.cntquick++;
      qh->qhmem.totfree -= outsize;
      *freelistp= *((void **)*freelistp);  /* replace freelist with next object */
#ifdef qh_TRACEshort
      n= qh->qhmem.cntshort+qh->qhmem.cntquick+qh->qhmem.freeshort;
      if (qh->qhmem.IStracing >= 5)
          qh_fprintf(qh, qh->qhmem.ferr, 8141, "qh_mem %p n %8d alloc quick: %d bytes (tot %d cnt %d)\n", object, n, outsize, qh->qhmem.totshort, qh->qhmem.cntshort+qh->qhmem.cntquick-qh->qhmem.freeshort);
#endif
      return(object);
    }else {
      qh->qhmem.cntshort++;
      if (outsize > qh->qhmem.freesize) {
        qh->qhmem.totdropped += qh->qhmem.freesize;
        if (!qh->qhmem.curbuffer)
          bufsize= qh->qhmem.BUFinit;
        else
          bufsize= qh->qhmem.BUFsize;
        if (!(newbuffer= qh_malloc((size_t)bufsize))) {
          qh_fprintf(qh, qh->qhmem.ferr, 6080, "qhull error (qh_memalloc): insufficient memory to allocate short memory buffer (%d bytes)\n", bufsize);
          qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
        }
        *((void **)newbuffer)= qh->qhmem.curbuffer;  /* prepend newbuffer to curbuffer
                                                    list.  newbuffer!=0 by QH6080 */
        qh->qhmem.curbuffer= newbuffer;
        size= ((int)sizeof(void **) + qh->qhmem.ALIGNmask) & ~qh->qhmem.ALIGNmask;
        qh->qhmem.freemem= (void *)((char *)newbuffer+size);
        qh->qhmem.freesize= bufsize - size;
        qh->qhmem.totbuffer += bufsize - size; /* easier to check */
        /* Periodically test totbuffer.  It matches at beginning and exit of every call */
        n= qh->qhmem.totshort + qh->qhmem.totfree + qh->qhmem.totdropped + qh->qhmem.freesize - outsize;
        if (qh->qhmem.totbuffer != n) {
            qh_fprintf(qh, qh->qhmem.ferr, 6212, "qhull internal error (qh_memalloc): short totbuffer %d != totshort+totfree... %d\n", qh->qhmem.totbuffer, n);
            qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
        }
      }
      object= qh->qhmem.freemem;
      qh->qhmem.freemem= (void *)((char *)qh->qhmem.freemem + outsize);
      qh->qhmem.freesize -= outsize;
      qh->qhmem.totunused += outsize - insize;
#ifdef qh_TRACEshort
      n= qh->qhmem.cntshort+qh->qhmem.cntquick+qh->qhmem.freeshort;
      if (qh->qhmem.IStracing >= 5)
          qh_fprintf(qh, qh->qhmem.ferr, 8140, "qh_mem %p n %8d alloc short: %d bytes (tot %d cnt %d)\n", object, n, outsize, qh->qhmem.totshort, qh->qhmem.cntshort+qh->qhmem.cntquick-qh->qhmem.freeshort);
#endif
      return object;
    }
  }else {                     /* long allocation */
    if (!qh->qhmem.indextable) {
      qh_fprintf(qh, qh->qhmem.ferr, 6081, "qhull internal error (qh_memalloc): qhmem has not been initialized.\n");
      qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
    }
    outsize= insize;
    qh->qhmem.cntlong++;
    qh->qhmem.totlong += outsize;
    if (qh->qhmem.maxlong < qh->qhmem.totlong)
      qh->qhmem.maxlong= qh->qhmem.totlong;
    if (!(object= qh_malloc((size_t)outsize))) {
      qh_fprintf(qh, qh->qhmem.ferr, 6082, "qhull error (qh_memalloc): insufficient memory to allocate %d bytes\n", outsize);
      qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
    }
    if (qh->qhmem.IStracing >= 5)
      qh_fprintf(qh, qh->qhmem.ferr, 8057, "qh_mem %p n %8d alloc long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, outsize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
  }
  return(object);
} /* memalloc */


/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memcheck">-</a>

  qh_memcheck(qh)
*/
void qh_memcheck(qhT *qh) {
  int i, count, totfree= 0;
  void *object;

  if (!qh) {
    qh_fprintf_stderr(6243, "qhull internal error (qh_memcheck): qh is 0.  It does not point to a qhT\n");
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  if (qh->qhmem.ferr == 0 || qh->qhmem.IStracing < 0 || qh->qhmem.IStracing > 10 || (((qh->qhmem.ALIGNmask+1) & qh->qhmem.ALIGNmask) != 0)) {
    qh_fprintf_stderr(6244, "qhull internal error (qh_memcheck): either qh->qhmem is overwritten or qh->qhmem is not initialized.  Call qh_meminit or qh_new_qhull before calling qh_mem routines.  ferr 0x%x, IsTracing %d, ALIGNmask 0x%x\n", 
          qh->qhmem.ferr, qh->qhmem.IStracing, qh->qhmem.ALIGNmask);
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  if (qh->qhmem.IStracing != 0)
    qh_fprintf(qh, qh->qhmem.ferr, 8143, "qh_memcheck: check size of freelists on qh->qhmem\nqh_memcheck: A segmentation fault indicates an overwrite of qh->qhmem\n");
  for (i=0; i < qh->qhmem.TABLEsize; i++) {
    count=0;
    for (object= qh->qhmem.freelists[i]; object; object= *((void **)object))
      count++;
    totfree += qh->qhmem.sizetable[i] * count;
  }
  if (totfree != qh->qhmem.totfree) {
    qh_fprintf(qh, qh->qhmem.ferr, 6211, "qhull internal error (qh_memcheck): totfree %d not equal to freelist total %d\n", qh->qhmem.totfree, totfree);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  if (qh->qhmem.IStracing != 0)
    qh_fprintf(qh, qh->qhmem.ferr, 8144, "qh_memcheck: total size of freelists totfree is the same as qh->qhmem.totfree\n", totfree);
} /* memcheck */

/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="memfree">-</a>

  qh_memfree(qh, object, insize )
    free up an object of size bytes
    size is insize from qh_memalloc

  notes:
    object may be NULL
    type checking warns if using (void **)object
    use qh_memfree_() for quick free's of small objects

  design:
    if size <= qh->qhmem.LASTsize
      append object to corresponding freelist
    else
      call qh_free(object)
*/
void qh_memfree(qhT *qh, void *object, int insize) {
  void **freelistp;
  int idx, outsize;

  if (!object)
    return;
  if (insize <= qh->qhmem.LASTsize) {
    qh->qhmem.freeshort++;
    idx= qh->qhmem.indextable[insize];
    outsize= qh->qhmem.sizetable[idx];
    qh->qhmem.totfree += outsize;
    qh->qhmem.totshort -= outsize;
    freelistp= qh->qhmem.freelists + idx;
    *((void **)object)= *freelistp;
    *freelistp= object;
#ifdef qh_TRACEshort
    idx= qh->qhmem.cntshort+qh->qhmem.cntquick+qh->qhmem.freeshort;
    if (qh->qhmem.IStracing >= 5)
        qh_fprintf(qh, qh->qhmem.ferr, 8142, "qh_mem %p n %8d free short: %d bytes (tot %d cnt %d)\n", object, idx, outsize, qh->qhmem.totshort, qh->qhmem.cntshort+qh->qhmem.cntquick-qh->qhmem.freeshort);
#endif
  }else {
    qh->qhmem.freelong++;
    qh->qhmem.totlong -= insize;
    if (qh->qhmem.IStracing >= 5)
      qh_fprintf(qh, qh->qhmem.ferr, 8058, "qh_mem %p n %8d free long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, insize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
    qh_free(object);
  }
} /* memfree */


/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memfreeshort">-</a>

  qh_memfreeshort(qh, curlong, totlong )
    frees up all short and qhmem memory allocations

  returns:
    number and size of current long allocations

  notes:
    if qh_NOmem (qh_malloc() for all allocations),
       short objects (e.g., facetT) are not recovered.
       use qh_freeqhull(qh, qh_ALL) instead.

  see:
    qh_freeqhull(qh, allMem)
    qh_memtotal(qh, curlong, totlong, curshort, totshort, maxlong, totbuffer);
*/
void qh_memfreeshort(qhT *qh, int *curlong, int *totlong) {
  void *buffer, *nextbuffer;
  FILE *ferr;

  *curlong= qh->qhmem.cntlong - qh->qhmem.freelong;
  *totlong= qh->qhmem.totlong;
  for (buffer=qh->qhmem.curbuffer; buffer; buffer= nextbuffer) {
    nextbuffer= *((void **) buffer);
    qh_free(buffer);
  }
  qh->qhmem.curbuffer= NULL;
  if (qh->qhmem.LASTsize) {
    qh_free(qh->qhmem.indextable);
    qh_free(qh->qhmem.freelists);
    qh_free(qh->qhmem.sizetable);
  }
  ferr= qh->qhmem.ferr;
  memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
  qh->qhmem.ferr= ferr;
} /* memfreeshort */


/*-<a                             href="qh-mem_r.htm#TOC"
  >--------------------------------</a><a name="meminit">-</a>

  qh_meminit(qh, ferr )
    initialize qhmem and test sizeof(void *)
    Does not throw errors.  qh_exit on failure
*/
void qh_meminit(qhT *qh, FILE *ferr) {

  memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
  if (ferr)
    qh->qhmem.ferr= ferr;
  else
    qh->qhmem.ferr= stderr;
  if (sizeof(void *) < sizeof(int)) {
    qh_fprintf(qh, qh->qhmem.ferr, 6083, "qhull internal error (qh_meminit): sizeof(void *) %d < sizeof(int) %d.  qset_r.c will not work\n", (int)sizeof(void*), (int)sizeof(int));
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  if (sizeof(void *) > sizeof(ptr_intT)) {
    qh_fprintf(qh, qh->qhmem.ferr, 6084, "qhull internal error (qh_meminit): sizeof(void *) %d > sizeof(ptr_intT) %d. Change ptr_intT in mem_r.h to 'long long'\n", (int)sizeof(void*), (int)sizeof(ptr_intT));
    qh_exit(qhmem_ERRqhull);  /* can not use qh_errexit() */
  }
  qh_memcheck(qh);
} /* meminit */

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="meminitbuffers">-</a>

  qh_meminitbuffers(qh, tracelevel, alignment, numsizes, bufsize, bufinit )
    initialize qhmem
    if tracelevel >= 5, trace memory allocations
    alignment= desired address alignment for memory allocations
    numsizes= number of freelists
    bufsize=  size of additional memory buffers for short allocations
    bufinit=  size of initial memory buffer for short allocations
*/
void qh_meminitbuffers(qhT *qh, int tracelevel, int alignment, int numsizes, int bufsize, int bufinit) {

  qh->qhmem.IStracing= tracelevel;
  qh->qhmem.NUMsizes= numsizes;
  qh->qhmem.BUFsize= bufsize;
  qh->qhmem.BUFinit= bufinit;
  qh->qhmem.ALIGNmask= alignment-1;
  if (qh->qhmem.ALIGNmask & ~qh->qhmem.ALIGNmask) {
    qh_fprintf(qh, qh->qhmem.ferr, 6085, "qhull internal error (qh_meminit): memory alignment %d is not a power of 2\n", alignment);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  qh->qhmem.sizetable= (int *) calloc((size_t)numsizes, sizeof(int));
  qh->qhmem.freelists= (void **) calloc((size_t)numsizes, sizeof(void *));
  if (!qh->qhmem.sizetable || !qh->qhmem.freelists) {
    qh_fprintf(qh, qh->qhmem.ferr, 6086, "qhull error (qh_meminit): insufficient memory\n");
    qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
  }
  if (qh->qhmem.IStracing >= 1)
    qh_fprintf(qh, qh->qhmem.ferr, 8059, "qh_meminitbuffers: memory initialized with alignment %d\n", alignment);
} /* meminitbuffers */

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memsetup">-</a>

  qh_memsetup(qh)
    set up memory after running memsize()
*/
void qh_memsetup(qhT *qh) {
  int k,i;

  qsort(qh->qhmem.sizetable, (size_t)qh->qhmem.TABLEsize, sizeof(int), qh_intcompare);
  qh->qhmem.LASTsize= qh->qhmem.sizetable[qh->qhmem.TABLEsize-1];
  if (qh->qhmem.LASTsize >= qh->qhmem.BUFsize || qh->qhmem.LASTsize >= qh->qhmem.BUFinit) {
    qh_fprintf(qh, qh->qhmem.ferr, 6087, "qhull error (qh_memsetup): largest mem size %d is >= buffer size %d or initial buffer size %d\n",
            qh->qhmem.LASTsize, qh->qhmem.BUFsize, qh->qhmem.BUFinit);
    qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
  }
  if (!(qh->qhmem.indextable= (int *)qh_malloc((size_t)(qh->qhmem.LASTsize+1) * sizeof(int)))) {
    qh_fprintf(qh, qh->qhmem.ferr, 6088, "qhull error (qh_memsetup): insufficient memory\n");
    qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
  }
  for (k=qh->qhmem.LASTsize+1; k--; )
    qh->qhmem.indextable[k]= k;
  i= 0;
  for (k=0; k <= qh->qhmem.LASTsize; k++) {
    if (qh->qhmem.indextable[k] <= qh->qhmem.sizetable[i])
      qh->qhmem.indextable[k]= i;
    else
      qh->qhmem.indextable[k]= ++i;
  }
} /* memsetup */

/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memsize">-</a>

  qh_memsize(qh, size )
    define a free list for this size
*/
void qh_memsize(qhT *qh, int size) {
  int k;

  if (qh->qhmem.LASTsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6089, "qhull internal error (qh_memsize): qh_memsize called after qh_memsetup\n");
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  size= (size + qh->qhmem.ALIGNmask) & ~qh->qhmem.ALIGNmask;
  if (qh->qhmem.IStracing >= 3)
    qh_fprintf(qh, qh->qhmem.ferr, 3078, "qh_memsize: quick memory of %d bytes\n", size);
  for (k=qh->qhmem.TABLEsize; k--; ) {
    if (qh->qhmem.sizetable[k] == size)
      return;
  }
  if (qh->qhmem.TABLEsize < qh->qhmem.NUMsizes)
    qh->qhmem.sizetable[qh->qhmem.TABLEsize++]= size;
  else
    qh_fprintf(qh, qh->qhmem.ferr, 7060, "qhull warning (qh_memsize): free list table has room for only %d sizes\n", qh->qhmem.NUMsizes);
} /* memsize */


/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="memstatistics">-</a>

  qh_memstatistics(qh, fp )
    print out memory statistics

    Verifies that qh->qhmem.totfree == sum of freelists
*/
void qh_memstatistics(qhT *qh, FILE *fp) {
  int i;
  int count;
  void *object;

  qh_memcheck(qh);
  qh_fprintf(qh, fp, 9278, "\nmemory statistics:\n\
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
           qh->qhmem.cntquick, qh->qhmem.cntshort, qh->qhmem.cntlong,
           qh->qhmem.freeshort, qh->qhmem.freelong,
           qh->qhmem.totshort, qh->qhmem.totfree,
           qh->qhmem.totdropped + qh->qhmem.freesize, qh->qhmem.totunused,
           qh->qhmem.maxlong, qh->qhmem.totlong, qh->qhmem.cntlong - qh->qhmem.freelong,
           qh->qhmem.totbuffer, qh->qhmem.BUFsize, qh->qhmem.BUFinit);
  if (qh->qhmem.cntlarger) {
    qh_fprintf(qh, fp, 9279, "%7d calls to qh_setlarger\n%7.2g     average copy size\n",
           qh->qhmem.cntlarger, ((double)qh->qhmem.totlarger)/(double)qh->qhmem.cntlarger);
    qh_fprintf(qh, fp, 9280, "  freelists(bytes->count):");
  }
  for (i=0; i < qh->qhmem.TABLEsize; i++) {
    count=0;
    for (object= qh->qhmem.freelists[i]; object; object= *((void **)object))
      count++;
    qh_fprintf(qh, fp, 9281, " %d->%d", qh->qhmem.sizetable[i], count);
  }
  qh_fprintf(qh, fp, 9282, "\n\n");
} /* memstatistics */


/*-<a                             href="qh-mem_r.htm#TOC"
  >-------------------------------</a><a name="NOmem">-</a>

  qh_NOmem
    turn off quick-fit memory allocation

  notes:
    uses qh_malloc() and qh_free() instead
*/
#else /* qh_NOmem */

void *qh_memalloc(qhT *qh, int insize) {
  void *object;

  if (!(object= qh_malloc((size_t)insize))) {
    qh_fprintf(qh, qh->qhmem.ferr, 6090, "qhull error (qh_memalloc): insufficient memory\n");
    qh_errexit(qh, qhmem_ERRmem, NULL, NULL);
  }
  qh->qhmem.cntlong++;
  qh->qhmem.totlong += insize;
  if (qh->qhmem.maxlong < qh->qhmem.totlong)
      qh->qhmem.maxlong= qh->qhmem.totlong;
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8060, "qh_mem %p n %8d alloc long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, insize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
  return object;
}

void qh_memcheck(qhT *qh) {
}

void qh_memfree(qhT *qh, void *object, int insize) {

  if (!object)
    return;
  qh_free(object);
  qh->qhmem.freelong++;
  qh->qhmem.totlong -= insize;
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8061, "qh_mem %p n %8d free long: %d bytes (tot %d cnt %d)\n", object, qh->qhmem.cntlong+qh->qhmem.freelong, insize, qh->qhmem.totlong, qh->qhmem.cntlong-qh->qhmem.freelong);
}

void qh_memfreeshort(qhT *qh, int *curlong, int *totlong) {
  *totlong= qh->qhmem.totlong;
  *curlong= qh->qhmem.cntlong - qh->qhmem.freelong;
  memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
}

void qh_meminit(qhT *qh, FILE *ferr) {

  memset((char *)&qh->qhmem, 0, sizeof(qh->qhmem));  /* every field is 0, FALSE, NULL */
  if (ferr)
      qh->qhmem.ferr= ferr;
  else
      qh->qhmem.ferr= stderr;
  if (sizeof(void *) < sizeof(int)) {
    qh_fprintf(qh, qh->qhmem.ferr, 6091, "qhull internal error (qh_meminit): sizeof(void *) %d < sizeof(int) %d.  qset_r.c will not work\n", (int)sizeof(void*), (int)sizeof(int));
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
}

void qh_meminitbuffers(qhT *qh, int tracelevel, int alignment, int numsizes, int bufsize, int bufinit) {

  qh->qhmem.IStracing= tracelevel;
}

void qh_memsetup(qhT *qh) {
}

void qh_memsize(qhT *qh, int size) {
}

void qh_memstatistics(qhT *qh, FILE *fp) {

  qh_fprintf(qh, fp, 9409, "\nmemory statistics:\n\
%7d long allocations\n\
%7d long frees\n\
%7d bytes of long memory allocated (max, except for input)\n\
%7d bytes of long memory in use (in %d pieces)\n",
           qh->qhmem.cntlong,
           qh->qhmem.freelong,
           qh->qhmem.maxlong, qh->qhmem.totlong, qh->qhmem.cntlong - qh->qhmem.freelong);
}

#endif /* qh_NOmem */

/*-<a                             href="qh-mem_r.htm#TOC"
>-------------------------------</a><a name="memtotlong">-</a>

  qh_memtotal(qh, totlong, curlong, totshort, curshort, maxlong, totbuffer )
    Return the total, allocated long and short memory

  returns:
    Returns the total current bytes of long and short allocations
    Returns the current count of long and short allocations
    Returns the maximum long memory and total short buffer (minus one link per buffer)
    Does not error (for deprecated UsingLibQhull.cpp in libqhullpcpp)
*/
void qh_memtotal(qhT *qh, int *totlong, int *curlong, int *totshort, int *curshort, int *maxlong, int *totbuffer) {
    *totlong= qh->qhmem.totlong;
    *curlong= qh->qhmem.cntlong - qh->qhmem.freelong;
    *totshort= qh->qhmem.totshort;
    *curshort= qh->qhmem.cntshort + qh->qhmem.cntquick - qh->qhmem.freeshort;
    *maxlong= qh->qhmem.maxlong;
    *totbuffer= qh->qhmem.totbuffer;
} /* memtotlong */

