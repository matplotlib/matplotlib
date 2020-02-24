/*<html><pre>  -<a                             href="qh-set_r.htm"
  >-------------------------------</a><a name="TOP">-</a>

   qset_r.c
   implements set manipulations needed for quickhull

   see qh-set_r.htm and qset_r.h

   Be careful of strict aliasing (two pointers of different types
   that reference the same location).  The last slot of a set is
   either the actual size of the set plus 1, or the NULL terminator
   of the set (i.e., setelemT).

   Only reference qh for qhmem or qhstat.  Otherwise the matching code in qset.c will bring in qhT

   Copyright (c) 1993-2019 The Geometry Center.
   $Id: //main/2019/qhull/src/libqhull_r/qset_r.c#7 $$Change: 2711 $
   $DateTime: 2019/06/27 22:34:56 $$Author: bbarber $
*/

#include "libqhull_r.h" /* for qhT and QHULL_CRTDBG */
#include "qset_r.h"
#include "mem_r.h"
#include <stdio.h>
#include <string.h>
/*** uncomment here and qhull_ra.h
     if string.h does not define memcpy()
#include <memory.h>
*/

#ifndef qhDEFlibqhull
typedef struct ridgeT ridgeT;
typedef struct facetT facetT;
void    qh_errexit(qhT *qh, int exitcode, facetT *, ridgeT *);
void    qh_fprintf(qhT *qh, FILE *fp, int msgcode, const char *fmt, ... );
#  ifdef _MSC_VER  /* Microsoft Visual C++ -- warning level 4 */
#  pragma warning( disable : 4127)  /* conditional expression is constant */
#  pragma warning( disable : 4706)  /* assignment within conditional function */
#  endif
#endif

/*=============== internal macros ===========================*/

/*============ functions in alphabetical order ===================*/

/*-<a                             href="qh-set_r.htm#TOC"
  >--------------------------------<a name="setaddnth">-</a>

  qh_setaddnth(qh, setp, nth, newelem )
    adds newelem as n'th element of sorted or unsorted *setp

  notes:
    *setp and newelem must be defined
    *setp may be a temp set
    nth=0 is first element
    errors if nth is out of bounds

  design:
    expand *setp if empty or full
    move tail of *setp up one
    insert newelem
*/
void qh_setaddnth(qhT *qh, setT **setp, int nth, void *newelem) {
  int oldsize, i;
  setelemT *sizep;          /* avoid strict aliasing */
  setelemT *oldp, *newp;

  if (!*setp || (sizep= SETsizeaddr_(*setp))->i==0) {
    qh_setlarger(qh, setp);
    sizep= SETsizeaddr_(*setp);
  }
  oldsize= sizep->i - 1;
  if (nth < 0 || nth > oldsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6171, "qhull internal error (qh_setaddnth): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", *setp);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  sizep->i++;
  oldp= (setelemT *)SETelemaddr_(*setp, oldsize, void);   /* NULL */
  newp= oldp+1;
  for (i=oldsize-nth+1; i--; )  /* move at least NULL  */
    (newp--)->p= (oldp--)->p;       /* may overwrite *sizep */
  newp->p= newelem;
} /* setaddnth */


/*-<a                              href="qh-set_r.htm#TOC"
  >--------------------------------<a name="setaddsorted">-</a>

  setaddsorted( setp, newelem )
    adds an newelem into sorted *setp

  notes:
    *setp and newelem must be defined
    *setp may be a temp set
    nop if newelem already in set

  design:
    find newelem's position in *setp
    insert newelem
*/
void qh_setaddsorted(qhT *qh, setT **setp, void *newelem) {
  int newindex=0;
  void *elem, **elemp;

  FOREACHelem_(*setp) {          /* could use binary search instead */
    if (elem < newelem)
      newindex++;
    else if (elem == newelem)
      return;
    else
      break;
  }
  qh_setaddnth(qh, setp, newindex, newelem);
} /* setaddsorted */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setappend">-</a>

  qh_setappend(qh, setp, newelem )
    append newelem to *setp

  notes:
    *setp may be a temp set
    *setp and newelem may be NULL

  design:
    expand *setp if empty or full
    append newelem to *setp

*/
void qh_setappend(qhT *qh, setT **setp, void *newelem) {
  setelemT *sizep;  /* Avoid strict aliasing.  Writing to *endp may overwrite *sizep */
  setelemT *endp;
  int count;

  if (!newelem)
    return;
  if (!*setp || (sizep= SETsizeaddr_(*setp))->i==0) {
    qh_setlarger(qh, setp);
    sizep= SETsizeaddr_(*setp);
  }
  count= (sizep->i)++ - 1;
  endp= (setelemT *)SETelemaddr_(*setp, count, void);
  (endp++)->p= newelem;
  endp->p= NULL;
} /* setappend */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setappend_set">-</a>

  qh_setappend_set(qh, setp, setA )
    appends setA to *setp

  notes:
    *setp can not be a temp set
    *setp and setA may be NULL

  design:
    setup for copy
    expand *setp if it is too small
    append all elements of setA to *setp
*/
void qh_setappend_set(qhT *qh, setT **setp, setT *setA) {
  int sizeA, size;
  setT *oldset;
  setelemT *sizep;

  if (!setA)
    return;
  SETreturnsize_(setA, sizeA);
  if (!*setp)
    *setp= qh_setnew(qh, sizeA);
  sizep= SETsizeaddr_(*setp);
  if (!(size= sizep->i))
    size= (*setp)->maxsize;
  else
    size--;
  if (size + sizeA > (*setp)->maxsize) {
    oldset= *setp;
    *setp= qh_setcopy(qh, oldset, sizeA);
    qh_setfree(qh, &oldset);
    sizep= SETsizeaddr_(*setp);
  }
  if (sizeA > 0) {
    sizep->i= size+sizeA+1;   /* memcpy may overwrite */
    memcpy((char *)&((*setp)->e[size].p), (char *)&(setA->e[0].p), (size_t)(sizeA+1) * SETelemsize);
  }
} /* setappend_set */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setappend2ndlast">-</a>

  qh_setappend2ndlast(qh, setp, newelem )
    makes newelem the next to the last element in *setp

  notes:
    *setp must have at least one element
    newelem must be defined
    *setp may be a temp set

  design:
    expand *setp if empty or full
    move last element of *setp up one
    insert newelem
*/
void qh_setappend2ndlast(qhT *qh, setT **setp, void *newelem) {
    setelemT *sizep;  /* Avoid strict aliasing.  Writing to *endp may overwrite *sizep */
    setelemT *endp, *lastp;
    int count;

    if (!*setp || (sizep= SETsizeaddr_(*setp))->i==0) {
        qh_setlarger(qh, setp);
        sizep= SETsizeaddr_(*setp);
    }
    count= (sizep->i)++ - 1;
    endp= (setelemT *)SETelemaddr_(*setp, count, void); /* NULL */
    lastp= endp-1;
    *(endp++)= *lastp;
    endp->p= NULL;    /* may overwrite *sizep */
    lastp->p= newelem;
} /* setappend2ndlast */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setcheck">-</a>

  qh_setcheck(qh, set, typename, id )
    check set for validity
    report errors with typename and id

  design:
    checks that maxsize, actual size, and NULL terminator agree
*/
void qh_setcheck(qhT *qh, setT *set, const char *tname, unsigned int id) {
  int maxsize, size;
  int waserr= 0;

  if (!set)
    return;
  SETreturnsize_(set, size);
  maxsize= set->maxsize;
  if (size > maxsize || !maxsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6172, "qhull internal error (qh_setcheck): actual size %d of %s%d is greater than max size %d\n",
             size, tname, id, maxsize);
    waserr= 1;
  }else if (set->e[size].p) {
    qh_fprintf(qh, qh->qhmem.ferr, 6173, "qhull internal error (qh_setcheck): %s%d(size %d max %d) is not null terminated.\n",
             tname, id, size-1, maxsize);
    waserr= 1;
  }
  if (waserr) {
    qh_setprint(qh, qh->qhmem.ferr, "ERRONEOUS", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
} /* setcheck */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setcompact">-</a>

  qh_setcompact(qh, set )
    remove internal NULLs from an unsorted set

  returns:
    updated set

  notes:
    set may be NULL
    it would be faster to swap tail of set into holes, like qh_setdel

  design:
    setup pointers into set
    skip NULLs while copying elements to start of set
    update the actual size
*/
void qh_setcompact(qhT *qh, setT *set) {
  int size;
  void **destp, **elemp, **endp, **firstp;

  if (!set)
    return;
  SETreturnsize_(set, size);
  destp= elemp= firstp= SETaddr_(set, void);
  endp= destp + size;
  while (1) {
    if (!(*destp++= *elemp++)) {
      destp--;
      if (elemp > endp)
        break;
    }
  }
  qh_settruncate(qh, set, (int)(destp-firstp));   /* WARN64 */
} /* setcompact */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setcopy">-</a>

  qh_setcopy(qh, set, extra )
    make a copy of a sorted or unsorted set with extra slots

  returns:
    new set

  design:
    create a newset with extra slots
    copy the elements to the newset

*/
setT *qh_setcopy(qhT *qh, setT *set, int extra) {
  setT *newset;
  int size;

  if (extra < 0)
    extra= 0;
  SETreturnsize_(set, size);
  newset= qh_setnew(qh, size+extra);
  SETsizeaddr_(newset)->i= size+1;    /* memcpy may overwrite */
  memcpy((char *)&(newset->e[0].p), (char *)&(set->e[0].p), (size_t)(size+1) * SETelemsize);
  return(newset);
} /* setcopy */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdel">-</a>

  qh_setdel(set, oldelem )
    delete oldelem from an unsorted set

  returns:
    returns oldelem if found
    returns NULL otherwise

  notes:
    set may be NULL
    oldelem must not be NULL;
    only deletes one copy of oldelem in set

  design:
    locate oldelem
    update actual size if it was full
    move the last element to the oldelem's location
*/
void *qh_setdel(setT *set, void *oldelem) {
  setelemT *sizep;
  setelemT *elemp;
  setelemT *lastp;

  if (!set)
    return NULL;
  elemp= (setelemT *)SETaddr_(set, void);
  while (elemp->p != oldelem && elemp->p)
    elemp++;
  if (elemp->p) {
    sizep= SETsizeaddr_(set);
    if (!(sizep->i)--)         /*  if was a full set */
      sizep->i= set->maxsize;  /*     *sizep= (maxsize-1)+ 1 */
    lastp= (setelemT *)SETelemaddr_(set, sizep->i-1, void);
    elemp->p= lastp->p;      /* may overwrite itself */
    lastp->p= NULL;
    return oldelem;
  }
  return NULL;
} /* setdel */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdellast">-</a>

  qh_setdellast( set )
    return last element of set or NULL

  notes:
    deletes element from set
    set may be NULL

  design:
    return NULL if empty
    if full set
      delete last element and set actual size
    else
      delete last element and update actual size
*/
void *qh_setdellast(setT *set) {
  int setsize;  /* actually, actual_size + 1 */
  int maxsize;
  setelemT *sizep;
  void *returnvalue;

  if (!set || !(set->e[0].p))
    return NULL;
  sizep= SETsizeaddr_(set);
  if ((setsize= sizep->i)) {
    returnvalue= set->e[setsize - 2].p;
    set->e[setsize - 2].p= NULL;
    sizep->i--;
  }else {
    maxsize= set->maxsize;
    returnvalue= set->e[maxsize - 1].p;
    set->e[maxsize - 1].p= NULL;
    sizep->i= maxsize;
  }
  return returnvalue;
} /* setdellast */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdelnth">-</a>

  qh_setdelnth(qh, set, nth )
    deletes nth element from unsorted set
    0 is first element

  returns:
    returns the element (needs type conversion)

  notes:
    errors if nth invalid

  design:
    setup points and check nth
    delete nth element and overwrite with last element
*/
void *qh_setdelnth(qhT *qh, setT *set, int nth) {
  void *elem;
  setelemT *sizep;
  setelemT *elemp, *lastp;

  sizep= SETsizeaddr_(set);
  if ((sizep->i--)==0)         /*  if was a full set */
    sizep->i= set->maxsize;    /*    *sizep= (maxsize-1)+ 1 */
  if (nth < 0 || nth >= sizep->i) {
    qh_fprintf(qh, qh->qhmem.ferr, 6174, "qhull internal error (qh_setdelnth): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  elemp= (setelemT *)SETelemaddr_(set, nth, void); /* nth valid by QH6174 */
  lastp= (setelemT *)SETelemaddr_(set, sizep->i-1, void);
  elem= elemp->p;
  elemp->p= lastp->p;      /* may overwrite itself */
  lastp->p= NULL;
  return elem;
} /* setdelnth */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdelnthsorted">-</a>

  qh_setdelnthsorted(qh, set, nth )
    deletes nth element from sorted set

  returns:
    returns the element (use type conversion)

  notes:
    errors if nth invalid

  see also:
    setnew_delnthsorted

  design:
    setup points and check nth
    copy remaining elements down one
    update actual size
*/
void *qh_setdelnthsorted(qhT *qh, setT *set, int nth) {
  void *elem;
  setelemT *sizep;
  setelemT *newp, *oldp;

  sizep= SETsizeaddr_(set);
  if (nth < 0 || (sizep->i && nth >= sizep->i-1) || nth >= set->maxsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6175, "qhull internal error (qh_setdelnthsorted): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  newp= (setelemT *)SETelemaddr_(set, nth, void);
  elem= newp->p;
  oldp= newp+1;
  while (((newp++)->p= (oldp++)->p))
    ; /* copy remaining elements and NULL */
  if ((sizep->i--)==0)         /*  if was a full set */
    sizep->i= set->maxsize;  /*     *sizep= (max size-1)+ 1 */
  return elem;
} /* setdelnthsorted */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setdelsorted">-</a>

  qh_setdelsorted( set, oldelem )
    deletes oldelem from sorted set

  returns:
    returns oldelem if it was deleted

  notes:
    set may be NULL

  design:
    locate oldelem in set
    copy remaining elements down one
    update actual size
*/
void *qh_setdelsorted(setT *set, void *oldelem) {
  setelemT *sizep;
  setelemT *newp, *oldp;

  if (!set)
    return NULL;
  newp= (setelemT *)SETaddr_(set, void);
  while(newp->p != oldelem && newp->p)
    newp++;
  if (newp->p) {
    oldp= newp+1;
    while (((newp++)->p= (oldp++)->p))
      ; /* copy remaining elements */
    sizep= SETsizeaddr_(set);
    if ((sizep->i--)==0)    /*  if was a full set */
      sizep->i= set->maxsize;  /*     *sizep= (max size-1)+ 1 */
    return oldelem;
  }
  return NULL;
} /* setdelsorted */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setduplicate">-</a>

  qh_setduplicate(qh, set, elemsize )
    duplicate a set of elemsize elements

  notes:
    use setcopy if retaining old elements

  design:
    create a new set
    for each elem of the old set
      create a newelem
      append newelem to newset
*/
setT *qh_setduplicate(qhT *qh, setT *set, int elemsize) {
  void          *elem, **elemp, *newElem;
  setT          *newSet;
  int           size;

  if (!(size= qh_setsize(qh, set)))
    return NULL;
  newSet= qh_setnew(qh, size);
  FOREACHelem_(set) {
    newElem= qh_memalloc(qh, elemsize);
    memcpy(newElem, elem, (size_t)elemsize);
    qh_setappend(qh, &newSet, newElem);
  }
  return newSet;
} /* setduplicate */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setendpointer">-</a>

  qh_setendpointer( set )
    Returns pointer to NULL terminator of a set's elements
    set can not be NULL

*/
void **qh_setendpointer(setT *set) {

  setelemT *sizep= SETsizeaddr_(set);
  int n= sizep->i;
  return (n ? &set->e[n-1].p : &sizep->p);
} /* qh_setendpointer */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setequal">-</a>

  qh_setequal( setA, setB )
    returns 1 if two sorted sets are equal, otherwise returns 0

  notes:
    either set may be NULL

  design:
    check size of each set
    setup pointers
    compare elements of each set
*/
int qh_setequal(setT *setA, setT *setB) {
  void **elemAp, **elemBp;
  int sizeA= 0, sizeB= 0;

  if (setA) {
    SETreturnsize_(setA, sizeA);
  }
  if (setB) {
    SETreturnsize_(setB, sizeB);
  }
  if (sizeA != sizeB)
    return 0;
  if (!sizeA)
    return 1;
  elemAp= SETaddr_(setA, void);
  elemBp= SETaddr_(setB, void);
  if (!memcmp((char *)elemAp, (char *)elemBp, (size_t)(sizeA * SETelemsize)))
    return 1;
  return 0;
} /* setequal */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setequal_except">-</a>

  qh_setequal_except( setA, skipelemA, setB, skipelemB )
    returns 1 if sorted setA and setB are equal except for skipelemA & B

  returns:
    false if either skipelemA or skipelemB are missing

  notes:
    neither set may be NULL

    if skipelemB is NULL,
      can skip any one element of setB

  design:
    setup pointers
    search for skipelemA, skipelemB, and mismatches
    check results
*/
int qh_setequal_except(setT *setA, void *skipelemA, setT *setB, void *skipelemB) {
  void **elemA, **elemB;
  int skip=0;

  elemA= SETaddr_(setA, void);
  elemB= SETaddr_(setB, void);
  while (1) {
    if (*elemA == skipelemA) {
      skip++;
      elemA++;
    }
    if (skipelemB) {
      if (*elemB == skipelemB) {
        skip++;
        elemB++;
      }
    }else if (*elemA != *elemB) {
      skip++;
      if (!(skipelemB= *elemB++))
        return 0;
    }
    if (!*elemA)
      break;
    if (*elemA++ != *elemB++)
      return 0;
  }
  if (skip != 2 || *elemB)
    return 0;
  return 1;
} /* setequal_except */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setequal_skip">-</a>

  qh_setequal_skip( setA, skipA, setB, skipB )
    returns 1 if sorted setA and setB are equal except for elements skipA & B

  returns:
    false if different size

  notes:
    neither set may be NULL

  design:
    setup pointers
    search for mismatches while skipping skipA and skipB
*/
int qh_setequal_skip(setT *setA, int skipA, setT *setB, int skipB) {
  void **elemA, **elemB, **skipAp, **skipBp;

  elemA= SETaddr_(setA, void);
  elemB= SETaddr_(setB, void);
  skipAp= SETelemaddr_(setA, skipA, void);
  skipBp= SETelemaddr_(setB, skipB, void);
  while (1) {
    if (elemA == skipAp)
      elemA++;
    if (elemB == skipBp)
      elemB++;
    if (!*elemA)
      break;
    if (*elemA++ != *elemB++)
      return 0;
  }
  if (*elemB)
    return 0;
  return 1;
} /* setequal_skip */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setfree">-</a>

  qh_setfree(qh, setp )
    frees the space occupied by a sorted or unsorted set

  returns:
    sets setp to NULL

  notes:
    set may be NULL

  design:
    free array
    free set
*/
void qh_setfree(qhT *qh, setT **setp) {
  int size;
  void **freelistp;  /* used if !qh_NOmem by qh_memfree_() */

  if (*setp) {
    size= (int)sizeof(setT) + ((*setp)->maxsize)*SETelemsize;
    if (size <= qh->qhmem.LASTsize) {
      qh_memfree_(qh, *setp, size, freelistp);
    }else
      qh_memfree(qh, *setp, size);
    *setp= NULL;
  }
} /* setfree */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setfree2">-</a>

  qh_setfree2(qh, setp, elemsize )
    frees the space occupied by a set and its elements

  notes:
    set may be NULL

  design:
    free each element
    free set
*/
void qh_setfree2(qhT *qh, setT **setp, int elemsize) {
  void          *elem, **elemp;

  FOREACHelem_(*setp)
    qh_memfree(qh, elem, elemsize);
  qh_setfree(qh, setp);
} /* setfree2 */



/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setfreelong">-</a>

  qh_setfreelong(qh, setp )
    frees a set only if it's in long memory

  returns:
    sets setp to NULL if it is freed

  notes:
    set may be NULL

  design:
    if set is large
      free it
*/
void qh_setfreelong(qhT *qh, setT **setp) {
  int size;

  if (*setp) {
    size= (int)sizeof(setT) + ((*setp)->maxsize)*SETelemsize;
    if (size > qh->qhmem.LASTsize) {
      qh_memfree(qh, *setp, size);
      *setp= NULL;
    }
  }
} /* setfreelong */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setin">-</a>

  qh_setin( set, setelem )
    returns 1 if setelem is in a set, 0 otherwise

  notes:
    set may be NULL or unsorted

  design:
    scans set for setelem
*/
int qh_setin(setT *set, void *setelem) {
  void *elem, **elemp;

  FOREACHelem_(set) {
    if (elem == setelem)
      return 1;
  }
  return 0;
} /* setin */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setindex">-</a>

  qh_setindex(set, atelem )
    returns the index of atelem in set.
    returns -1, if not in set or maxsize wrong

  notes:
    set may be NULL and may contain nulls.
    NOerrors returned (qh_pointid, QhullPoint::id)

  design:
    checks maxsize
    scans set for atelem
*/
int qh_setindex(setT *set, void *atelem) {
  void **elem;
  int size, i;

  if (!set)
    return -1;
  SETreturnsize_(set, size);
  if (size > set->maxsize)
    return -1;
  elem= SETaddr_(set, void);
  for (i=0; i < size; i++) {
    if (*elem++ == atelem)
      return i;
  }
  return -1;
} /* setindex */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setlarger">-</a>

  qh_setlarger(qh, oldsetp )
    returns a larger set that contains all elements of *oldsetp

  notes:
    if long memory,
      the new set is 2x larger
    if qhmem.LASTsize is between 1.5x and 2x
      the new set is qhmem.LASTsize
    otherwise use quick memory,
      the new set is 2x larger, rounded up to next qh_memsize
       
    if temp set, updates qh->qhmem.tempstack

  design:
    creates a new set
    copies the old set to the new set
    updates pointers in tempstack
    deletes the old set
*/
void qh_setlarger(qhT *qh, setT **oldsetp) {
  int setsize= 1, newsize;
  setT *newset, *set, **setp, *oldset;
  setelemT *sizep;
  setelemT *newp, *oldp;

  if (*oldsetp) {
    oldset= *oldsetp;
    SETreturnsize_(oldset, setsize);
    qh->qhmem.cntlarger++;
    qh->qhmem.totlarger += setsize+1;
    qh_setlarger_quick(qh, setsize, &newsize);
    newset= qh_setnew(qh, newsize);
    oldp= (setelemT *)SETaddr_(oldset, void);
    newp= (setelemT *)SETaddr_(newset, void);
    memcpy((char *)newp, (char *)oldp, (size_t)(setsize+1) * SETelemsize);
    sizep= SETsizeaddr_(newset);
    sizep->i= setsize+1;
    FOREACHset_((setT *)qh->qhmem.tempstack) {
      if (set == oldset)
        *(setp-1)= newset;
    }
    qh_setfree(qh, oldsetp);
  }else
    newset= qh_setnew(qh, 3);
  *oldsetp= newset;
} /* setlarger */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setlarger_quick">-</a>

  qh_setlarger_quick(qh, setsize, newsize )
    determine newsize for setsize
    returns True if newsize fits in quick memory

  design:
    if 2x fits into quick memory
      return True, 2x
    if x+4 does not fit into quick memory
      return False, 2x
    if x+x/3 fits into quick memory
      return True, the last quick set
    otherwise
      return False, 2x
*/
int qh_setlarger_quick(qhT *qh, int setsize, int *newsize) {
    int lastquickset;

    *newsize= 2 * setsize;
    lastquickset= (qh->qhmem.LASTsize - (int)sizeof(setT)) / SETelemsize; /* matches size computation in qh_setnew */
    if (*newsize <= lastquickset)
      return 1;
    if (setsize + 4 > lastquickset)
      return 0;
    if (setsize + setsize/3 <= lastquickset) {
      *newsize= lastquickset;
      return 1;
    }
    return 0;
} /* setlarger_quick */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setlast">-</a>

  qh_setlast( set )
    return last element of set or NULL (use type conversion)

  notes:
    set may be NULL

  design:
    return last element
*/
void *qh_setlast(setT *set) {
  int size;

  if (set) {
    size= SETsizeaddr_(set)->i;
    if (!size)
      return SETelem_(set, set->maxsize - 1);
    else if (size > 1)
      return SETelem_(set, size - 2);
  }
  return NULL;
} /* setlast */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setnew">-</a>

  qh_setnew(qh, setsize )
    creates and allocates space for a set

  notes:
    setsize means the number of elements (!including the NULL terminator)
    use qh_settemp/qh_setfreetemp if set is temporary

  design:
    allocate memory for set
    roundup memory if small set
    initialize as empty set
*/
setT *qh_setnew(qhT *qh, int setsize) {
  setT *set;
  int sizereceived; /* used if !qh_NOmem */
  int size;
  void **freelistp; /* used if !qh_NOmem by qh_memalloc_() */

  if (!setsize)
    setsize++;
  size= (int)sizeof(setT) + setsize * SETelemsize; /* setT includes NULL terminator, see qh.LASTquickset */
  if (size>0 && size <= qh->qhmem.LASTsize) {
    qh_memalloc_(qh, size, freelistp, set, setT);
#ifndef qh_NOmem
    sizereceived= qh->qhmem.sizetable[ qh->qhmem.indextable[size]];
    if (sizereceived > size)
      setsize += (sizereceived - size)/SETelemsize;
#endif
  }else
    set= (setT *)qh_memalloc(qh, size);
  set->maxsize= setsize;
  set->e[setsize].i= 1;
  set->e[0].p= NULL;
  return(set);
} /* setnew */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setnew_delnthsorted">-</a>

  qh_setnew_delnthsorted(qh, set, size, nth, prepend )
    creates a sorted set not containing nth element
    if prepend, the first prepend elements are undefined

  notes:
    set must be defined
    checks nth
    see also: setdelnthsorted

  design:
    create new set
    setup pointers and allocate room for prepend'ed entries
    append head of old set to new set
    append tail of old set to new set
*/
setT *qh_setnew_delnthsorted(qhT *qh, setT *set, int size, int nth, int prepend) {
  setT *newset;
  void **oldp, **newp;
  int tailsize= size - nth -1, newsize;

  if (tailsize < 0) {
    qh_fprintf(qh, qh->qhmem.ferr, 6176, "qhull internal error (qh_setnew_delnthsorted): nth %d is out-of-bounds for set:\n", nth);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  newsize= size-1 + prepend;
  newset= qh_setnew(qh, newsize);
  newset->e[newset->maxsize].i= newsize+1;  /* may be overwritten */
  oldp= SETaddr_(set, void);
  newp= SETaddr_(newset, void) + prepend;
  switch (nth) {
  case 0:
    break;
  case 1:
    *(newp++)= *oldp++;
    break;
  case 2:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  case 3:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  case 4:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  default:
    memcpy((char *)newp, (char *)oldp, (size_t)nth * SETelemsize);
    newp += nth;
    oldp += nth;
    break;
  }
  oldp++;
  switch (tailsize) {
  case 0:
    break;
  case 1:
    *(newp++)= *oldp++;
    break;
  case 2:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  case 3:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  case 4:
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    *(newp++)= *oldp++;
    break;
  default:
    memcpy((char *)newp, (char *)oldp, (size_t)tailsize * SETelemsize);
    newp += tailsize;
  }
  *newp= NULL;
  return(newset);
} /* setnew_delnthsorted */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setprint">-</a>

  qh_setprint(qh, fp, string, set )
    print set elements to fp with identifying string

  notes:
    never errors
*/
void qh_setprint(qhT *qh, FILE *fp, const char* string, setT *set) {
  int size, k;

  if (!set)
    qh_fprintf(qh, fp, 9346, "%s set is null\n", string);
  else {
    SETreturnsize_(set, size);
    qh_fprintf(qh, fp, 9347, "%s set=%p maxsize=%d size=%d elems=",
             string, set, set->maxsize, size);
    if (size > set->maxsize)
      size= set->maxsize+1;
    for (k=0; k < size; k++)
      qh_fprintf(qh, fp, 9348, " %p", set->e[k].p);
    qh_fprintf(qh, fp, 9349, "\n");
  }
} /* setprint */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setreplace">-</a>

  qh_setreplace(qh, set, oldelem, newelem )
    replaces oldelem in set with newelem

  notes:
    errors if oldelem not in the set
    newelem may be NULL, but it turns the set into an indexed set (no FOREACH)

  design:
    find oldelem
    replace with newelem
*/
void qh_setreplace(qhT *qh, setT *set, void *oldelem, void *newelem) {
  void **elemp;

  elemp= SETaddr_(set, void);
  while (*elemp != oldelem && *elemp)
    elemp++;
  if (*elemp)
    *elemp= newelem;
  else {
    qh_fprintf(qh, qh->qhmem.ferr, 6177, "qhull internal error (qh_setreplace): elem %p not found in set\n",
       oldelem);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
} /* setreplace */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setsize">-</a>

  qh_setsize(qh, set )
    returns the size of a set

  notes:
    errors if set's maxsize is incorrect
    same as SETreturnsize_(set)
    same code for qh_setsize [qset_r.c] and QhullSetBase::count
    if first element is NULL, SETempty_() is True but qh_setsize may be greater than 0

  design:
    determine actual size of set from maxsize
*/
int qh_setsize(qhT *qh, setT *set) {
  int size;
  setelemT *sizep;

  if (!set)
    return(0);
  sizep= SETsizeaddr_(set);
  if ((size= sizep->i)) {
    size--;
    if (size > set->maxsize) {
      qh_fprintf(qh, qh->qhmem.ferr, 6178, "qhull internal error (qh_setsize): current set size %d is greater than maximum size %d\n",
               size, set->maxsize);
      qh_setprint(qh, qh->qhmem.ferr, "set: ", set);
      qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
    }
  }else
    size= set->maxsize;
  return size;
} /* setsize */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settemp">-</a>

  qh_settemp(qh, setsize )
    return a stacked, temporary set of up to setsize elements

  notes:
    use settempfree or settempfree_all to release from qh->qhmem.tempstack
    see also qh_setnew

  design:
    allocate set
    append to qh->qhmem.tempstack

*/
setT *qh_settemp(qhT *qh, int setsize) {
  setT *newset;

  newset= qh_setnew(qh, setsize);
  qh_setappend(qh, &qh->qhmem.tempstack, newset);
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8123, "qh_settemp: temp set %p of %d elements, depth %d\n",
       newset, newset->maxsize, qh_setsize(qh, qh->qhmem.tempstack));
  return newset;
} /* settemp */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settempfree">-</a>

  qh_settempfree(qh, set )
    free temporary set at top of qh->qhmem.tempstack

  notes:
    nop if set is NULL
    errors if set not from previous   qh_settemp

  to locate errors:
    use 'T2' to find source and then find mis-matching qh_settemp

  design:
    check top of qh->qhmem.tempstack
    free it
*/
void qh_settempfree(qhT *qh, setT **set) {
  setT *stackedset;

  if (!*set)
    return;
  stackedset= qh_settemppop(qh);
  if (stackedset != *set) {
    qh_settemppush(qh, stackedset);
    qh_fprintf(qh, qh->qhmem.ferr, 6179, "qhull internal error (qh_settempfree): set %p(size %d) was not last temporary allocated(depth %d, set %p, size %d)\n",
             *set, qh_setsize(qh, *set), qh_setsize(qh, qh->qhmem.tempstack)+1,
             stackedset, qh_setsize(qh, stackedset));
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  qh_setfree(qh, set);
} /* settempfree */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settempfree_all">-</a>

  qh_settempfree_all(qh)
    free all temporary sets in qh->qhmem.tempstack

  design:
    for each set in tempstack
      free set
    free qh->qhmem.tempstack
*/
void qh_settempfree_all(qhT *qh) {
  setT *set, **setp;

  FOREACHset_(qh->qhmem.tempstack)
    qh_setfree(qh, &set);
  qh_setfree(qh, &qh->qhmem.tempstack);
} /* settempfree_all */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settemppop">-</a>

  qh_settemppop(qh)
    pop and return temporary set from qh->qhmem.tempstack

  notes:
    the returned set is permanent

  design:
    pop and check top of qh->qhmem.tempstack
*/
setT *qh_settemppop(qhT *qh) {
  setT *stackedset;

  stackedset= (setT *)qh_setdellast(qh->qhmem.tempstack);
  if (!stackedset) {
    qh_fprintf(qh, qh->qhmem.ferr, 6180, "qhull internal error (qh_settemppop): pop from empty temporary stack\n");
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8124, "qh_settemppop: depth %d temp set %p of %d elements\n",
       qh_setsize(qh, qh->qhmem.tempstack)+1, stackedset, qh_setsize(qh, stackedset));
  return stackedset;
} /* settemppop */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settemppush">-</a>

  qh_settemppush(qh, set )
    push temporary set unto qh->qhmem.tempstack (makes it temporary)

  notes:
    duplicates settemp() for tracing

  design:
    append set to tempstack
*/
void qh_settemppush(qhT *qh, setT *set) {
  if (!set) {
    qh_fprintf(qh, qh->qhmem.ferr, 6267, "qhull error (qh_settemppush): can not push a NULL temp\n");
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  qh_setappend(qh, &qh->qhmem.tempstack, set);
  if (qh->qhmem.IStracing >= 5)
    qh_fprintf(qh, qh->qhmem.ferr, 8125, "qh_settemppush: depth %d temp set %p of %d elements\n",
      qh_setsize(qh, qh->qhmem.tempstack), set, qh_setsize(qh, set));
} /* settemppush */


/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="settruncate">-</a>

  qh_settruncate(qh, set, size )
    truncate set to size elements

  notes:
    set must be defined

  see:
    SETtruncate_

  design:
    check size
    update actual size of set
*/
void qh_settruncate(qhT *qh, setT *set, int size) {

  if (size < 0 || size > set->maxsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6181, "qhull internal error (qh_settruncate): size %d out of bounds for set:\n", size);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  set->e[set->maxsize].i= size+1;   /* maybe overwritten */
  set->e[size].p= NULL;
} /* settruncate */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setunique">-</a>

  qh_setunique(qh, set, elem )
    add elem to unsorted set unless it is already in set

  notes:
    returns 1 if it is appended

  design:
    if elem not in set
      append elem to set
*/
int qh_setunique(qhT *qh, setT **set, void *elem) {

  if (!qh_setin(*set, elem)) {
    qh_setappend(qh, set, elem);
    return 1;
  }
  return 0;
} /* setunique */

/*-<a                             href="qh-set_r.htm#TOC"
  >-------------------------------<a name="setzero">-</a>

  qh_setzero(qh, set, index, size )
    zero elements from index on
    set actual size of set to size

  notes:
    set must be defined
    the set becomes an indexed set (can not use FOREACH...)

  see also:
    qh_settruncate

  design:
    check index and size
    update actual size
    zero elements starting at e[index]
*/
void qh_setzero(qhT *qh, setT *set, int idx, int size) {
  int count;

  if (idx < 0 || idx >= size || size > set->maxsize) {
    qh_fprintf(qh, qh->qhmem.ferr, 6182, "qhull internal error (qh_setzero): index %d or size %d out of bounds for set:\n", idx, size);
    qh_setprint(qh, qh->qhmem.ferr, "", set);
    qh_errexit(qh, qhmem_ERRqhull, NULL, NULL);
  }
  set->e[set->maxsize].i=  size+1;  /* may be overwritten */
  count= size - idx + 1;   /* +1 for NULL terminator */
  memset((char *)SETelemaddr_(set, idx, void), 0, (size_t)count * SETelemsize);
} /* setzero */


