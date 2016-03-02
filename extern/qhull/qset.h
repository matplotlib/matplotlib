/*<html><pre>  -<a                             href="qh-set.htm"
  >-------------------------------</a><a name="TOP">-</a>

   qset.h
     header file for qset.c that implements set

   see qh-set.htm and qset.c

   only uses mem.c, malloc/free

   for error handling, writes message and calls
      qh_errexit(qhmem_ERRqhull, NULL, NULL);

   set operations satisfy the following properties:
    - sets have a max size, the actual size (if different) is stored at the end
    - every set is NULL terminated
    - sets may be sorted or unsorted, the caller must distinguish this

   Copyright (c) 1993-2012 The Geometry Center.
   $Id: //main/2011/qhull/src/libqhull/qset.h#4 $$Change: 1464 $
   $DateTime: 2012/01/25 22:58:41 $$Author: bbarber $
*/

#ifndef qhDEFset
#define qhDEFset 1

#include <stdio.h>

/*================= -structures- ===============*/

#ifndef DEFsetT
#define DEFsetT 1
typedef struct setT setT;   /* a set is a sorted or unsorted array of pointers */
#endif

/*-<a                                      href="qh-set.htm#TOC"
>----------------------------------------</a><a name="setT">-</a>

setT
  a set or list of pointers with maximum size and actual size.

variations:
  unsorted, unique   -- a list of unique pointers with NULL terminator
                           user guarantees uniqueness
  sorted             -- a sorted list of unique pointers with NULL terminator
                           qset.c guarantees uniqueness
  unsorted           -- a list of pointers terminated with NULL
  indexed            -- an array of pointers with NULL elements

structure for set of n elements:

        --------------
        |  maxsize
        --------------
        |  e[0] - a pointer, may be NULL for indexed sets
        --------------
        |  e[1]

        --------------
        |  ...
        --------------
        |  e[n-1]
        --------------
        |  e[n] = NULL
        --------------
        |  ...
        --------------
        |  e[maxsize] - n+1 or NULL (determines actual size of set)
        --------------

*/

/*-- setelemT -- internal type to allow both pointers and indices
*/
typedef union setelemT setelemT;
union setelemT {
  void    *p;
  int      i;         /* integer used for e[maxSize] */
};

struct setT {
  int maxsize;          /* maximum number of elements (except NULL) */
  setelemT e[1];        /* array of pointers, tail is NULL */
                        /* last slot (unless NULL) is actual size+1
                           e[maxsize]==NULL or e[e[maxsize]-1]==NULL */
                        /* this may generate a warning since e[] contains
                           maxsize elements */
};

/*=========== -constants- =========================*/

/*-<a                                 href="qh-set.htm#TOC"
  >-----------------------------------</a><a name="SETelemsize">-</a>

  SETelemsize
    size of a set element in bytes
*/
#define SETelemsize ((int)sizeof(setelemT))


/*=========== -macros- =========================*/

/*-<a                                 href="qh-set.htm#TOC"
  >-----------------------------------</a><a name="FOREACHsetelement_">-</a>

   FOREACHsetelement_(type, set, variable)
     define FOREACH iterator

   declare:
     assumes *variable and **variablep are declared
     no space in "variable)" [DEC Alpha cc compiler]

   each iteration:
     variable is set element
     variablep is one beyond variable.

   to repeat an element:
     variablep--; / *repeat* /

   at exit:
     variable is NULL at end of loop

   example:
     #define FOREACHfacet_( facets ) FOREACHsetelement_( facetT, facets, facet )

   notes:
     use FOREACHsetelement_i_() if need index or include NULLs

   WARNING:
     nested loops can't use the same variable (define another FOREACH)

     needs braces if nested inside another FOREACH
     this includes intervening blocks, e.g. FOREACH...{ if () FOREACH...} )
*/
#define FOREACHsetelement_(type, set, variable) \
        if (((variable= NULL), set)) for (\
          variable##p= (type **)&((set)->e[0].p); \
          (variable= *variable##p++);)

/*-<a                                      href="qh-set.htm#TOC"
  >----------------------------------------</a><a name="FOREACHsetelement_i_">-</a>

   FOREACHsetelement_i_(type, set, variable)
     define indexed FOREACH iterator

   declare:
     type *variable, variable_n, variable_i;

   each iteration:
     variable is set element, may be NULL
     variable_i is index, variable_n is qh_setsize()

   to repeat an element:
     variable_i--; variable_n-- repeats for deleted element

   at exit:
     variable==NULL and variable_i==variable_n

   example:
     #define FOREACHfacet_i_( facets ) FOREACHsetelement_i_( facetT, facets, facet )

   WARNING:
     nested loops can't use the same variable (define another FOREACH)

     needs braces if nested inside another FOREACH
     this includes intervening blocks, e.g. FOREACH...{ if () FOREACH...} )
*/
#define FOREACHsetelement_i_(type, set, variable) \
        if (((variable= NULL), set)) for (\
          variable##_i= 0, variable= (type *)((set)->e[0].p), \
                   variable##_n= qh_setsize(set);\
          variable##_i < variable##_n;\
          variable= (type *)((set)->e[++variable##_i].p) )

/*-<a                                    href="qh-set.htm#TOC"
  >--------------------------------------</a><a name="FOREACHsetelementreverse_">-</a>

   FOREACHsetelementreverse_(type, set, variable)-
     define FOREACH iterator in reverse order

   declare:
     assumes *variable and **variablep are declared
     also declare 'int variabletemp'

   each iteration:
     variable is set element

   to repeat an element:
     variabletemp++; / *repeat* /

   at exit:
     variable is NULL

   example:
     #define FOREACHvertexreverse_( vertices ) FOREACHsetelementreverse_( vertexT, vertices, vertex )

   notes:
     use FOREACHsetelementreverse12_() to reverse first two elements
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHsetelementreverse_(type, set, variable) \
        if (((variable= NULL), set)) for (\
           variable##temp= qh_setsize(set)-1, variable= qh_setlast(set);\
           variable; variable= \
           ((--variable##temp >= 0) ? SETelemt_(set, variable##temp, type) : NULL))

/*-<a                                 href="qh-set.htm#TOC"
  >-----------------------------------</a><a name="FOREACHsetelementreverse12_">-</a>

   FOREACHsetelementreverse12_(type, set, variable)-
     define FOREACH iterator with e[1] and e[0] reversed

   declare:
     assumes *variable and **variablep are declared

   each iteration:
     variable is set element
     variablep is one after variable.

   to repeat an element:
     variablep--; / *repeat* /

   at exit:
     variable is NULL at end of loop

   example
     #define FOREACHvertexreverse12_( vertices ) FOREACHsetelementreverse12_( vertexT, vertices, vertex )

   notes:
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHsetelementreverse12_(type, set, variable) \
        if (((variable= NULL), set)) for (\
          variable##p= (type **)&((set)->e[1].p); \
          (variable= *variable##p); \
          variable##p == ((type **)&((set)->e[0].p))?variable##p += 2: \
              (variable##p == ((type **)&((set)->e[1].p))?variable##p--:variable##p++))

/*-<a                                 href="qh-set.htm#TOC"
  >-----------------------------------</a><a name="FOREACHelem_">-</a>

   FOREACHelem_( set )-
     iterate elements in a set

   declare:
     void *elem, *elemp;

   each iteration:
     elem is set element
     elemp is one beyond

   to repeat an element:
     elemp--; / *repeat* /

   at exit:
     elem == NULL at end of loop

   example:
     FOREACHelem_(set) {

   notes:
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHelem_(set) FOREACHsetelement_(void, set, elem)

/*-<a                                 href="qh-set.htm#TOC"
  >-----------------------------------</a><a name="FOREACHset_">-</a>

   FOREACHset_( set )-
     iterate a set of sets

   declare:
     setT *set, **setp;

   each iteration:
     set is set element
     setp is one beyond

   to repeat an element:
     setp--; / *repeat* /

   at exit:
     set == NULL at end of loop

   example
     FOREACHset_(sets) {

   notes:
     WARNING: needs braces if nested inside another FOREACH
*/
#define FOREACHset_(sets) FOREACHsetelement_(setT, sets, set)

/*-<a                                       href="qh-set.htm#TOC"
  >-----------------------------------------</a><a name="SETindex_">-</a>

   SETindex_( set, elem )
     return index of elem in set

   notes:
     for use with FOREACH iteration
     WARN64 -- Maximum set size is 2G

   example:
     i= SETindex_(ridges, ridge)
*/
#define SETindex_(set, elem) ((int)((void **)elem##p - (void **)&(set)->e[1].p))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETref_">-</a>

   SETref_( elem )
     l.h.s. for modifying the current element in a FOREACH iteration

   example:
     SETref_(ridge)= anotherridge;
*/
#define SETref_(elem) (elem##p[-1])

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETelem_">-</a>

   SETelem_(set, n)
     return the n'th element of set

   notes:
      assumes that n is valid [0..size] and that set is defined
      use SETelemt_() for type cast
*/
#define SETelem_(set, n)           ((set)->e[n].p)

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETelemt_">-</a>

   SETelemt_(set, n, type)
     return the n'th element of set as a type

   notes:
      assumes that n is valid [0..size] and that set is defined
*/
#define SETelemt_(set, n, type)    ((type*)((set)->e[n].p))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETelemaddr_">-</a>

   SETelemaddr_(set, n, type)
     return address of the n'th element of a set

   notes:
      assumes that n is valid [0..size] and set is defined
*/
#define SETelemaddr_(set, n, type) ((type **)(&((set)->e[n].p)))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETfirst_">-</a>

   SETfirst_(set)
     return first element of set

*/
#define SETfirst_(set)             ((set)->e[0].p)

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETfirstt_">-</a>

   SETfirstt_(set, type)
     return first element of set as a type

*/
#define SETfirstt_(set, type)      ((type*)((set)->e[0].p))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETsecond_">-</a>

   SETsecond_(set)
     return second element of set

*/
#define SETsecond_(set)            ((set)->e[1].p)

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETsecondt_">-</a>

   SETsecondt_(set, type)
     return second element of set as a type
*/
#define SETsecondt_(set, type)     ((type*)((set)->e[1].p))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETaddr_">-</a>

   SETaddr_(set, type)
       return address of set's elements
*/
#define SETaddr_(set,type)         ((type **)(&((set)->e[0].p)))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETreturnsize_">-</a>

   SETreturnsize_(set, size)
     return size of a set

   notes:
      set must be defined
      use qh_setsize(set) unless speed is critical
*/
#define SETreturnsize_(set, size) (((size)= ((set)->e[(set)->maxsize].i))?(--(size)):((size)= (set)->maxsize))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETempty_">-</a>

   SETempty_(set)
     return true(1) if set is empty

   notes:
      set may be NULL
*/
#define SETempty_(set)            (!set || (SETfirst_(set) ? 0 : 1))

/*-<a                             href="qh-set.htm#TOC"
  >-------------------------------<a name="SETsizeaddr_">-</a>

  SETsizeaddr_(set)
    return pointer to 'actual size+1' of set (set CANNOT be NULL!!)
    Its type is setelemT* for strict aliasing
    All SETelemaddr_ must be cast to setelemT


  notes:
    *SETsizeaddr==NULL or e[*SETsizeaddr-1].p==NULL
*/
#define SETsizeaddr_(set) (&((set)->e[(set)->maxsize]))

/*-<a                                     href="qh-set.htm#TOC"
  >---------------------------------------</a><a name="SETtruncate_">-</a>

   SETtruncate_(set, size)
     truncate set to size

   see:
     qh_settruncate()

*/
#define SETtruncate_(set, size) {set->e[set->maxsize].i= size+1; /* maybe overwritten */ \
      set->e[size].p= NULL;}

/*======= prototypes in alphabetical order ============*/

void  qh_setaddsorted(setT **setp, void *elem);
void  qh_setaddnth(setT **setp, int nth, void *newelem);
void  qh_setappend(setT **setp, void *elem);
void  qh_setappend_set(setT **setp, setT *setA);
void  qh_setappend2ndlast(setT **setp, void *elem);
void  qh_setcheck(setT *set, const char *tname, unsigned id);
void  qh_setcompact(setT *set);
setT *qh_setcopy(setT *set, int extra);
void *qh_setdel(setT *set, void *elem);
void *qh_setdellast(setT *set);
void *qh_setdelnth(setT *set, int nth);
void *qh_setdelnthsorted(setT *set, int nth);
void *qh_setdelsorted(setT *set, void *newelem);
setT *qh_setduplicate( setT *set, int elemsize);
int   qh_setequal(setT *setA, setT *setB);
int   qh_setequal_except(setT *setA, void *skipelemA, setT *setB, void *skipelemB);
int   qh_setequal_skip(setT *setA, int skipA, setT *setB, int skipB);
void **qh_setendpointer(setT *set);
void  qh_setfree(setT **set);
void  qh_setfree2( setT **setp, int elemsize);
void  qh_setfreelong(setT **set);
int   qh_setin(setT *set, void *setelem);
int   qh_setindex(setT *set, void *setelem);
void  qh_setlarger(setT **setp);
void *qh_setlast(setT *set);
setT *qh_setnew(int size);
setT *qh_setnew_delnthsorted(setT *set, int size, int nth, int prepend);
void  qh_setprint(FILE *fp, const char* string, setT *set);
void  qh_setreplace(setT *set, void *oldelem, void *newelem);
int   qh_setsize(setT *set);
setT *qh_settemp(int setsize);
void  qh_settempfree(setT **set);
void  qh_settempfree_all(void);
setT *qh_settemppop(void);
void  qh_settemppush(setT *set);
void  qh_settruncate(setT *set, int size);
int   qh_setunique(setT **set, void *elem);
void  qh_setzero(setT *set, int idx, int size);


#endif /* qhDEFset */
