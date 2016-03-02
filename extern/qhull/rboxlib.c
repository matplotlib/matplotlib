/*<html><pre>  -<a                             href="index.htm#TOC"
  >-------------------------------</a><a name="TOP">-</a>

   rboxlib.c
     Generate input points

   notes:
     For documentation, see prompt[] of rbox.c
     50 points generated for 'rbox D4'

   WARNING:
     incorrect range if qh_RANDOMmax is defined wrong (user.h)
*/

#include "random.h"
#include "libqhull.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER  /* Microsoft Visual C++ */
#pragma warning( disable : 4706)  /* assignment within conditional expression. */
#pragma warning( disable : 4996)  /* this function (strncat,sprintf,strcpy) or variable may be unsafe. */
#endif

#define MAXdim 200
#define PI 3.1415926535897932384

/* ------------------------------ prototypes ----------------*/
int roundi( double a);
void out1( double a);
void out2n( double a, double b);
void out3n( double a, double b, double c);

void    qh_fprintf_rbox(FILE *fp, int msgcode, const char *fmt, ... );
void    qh_free(void *mem);
void   *qh_malloc(size_t size);
int     qh_rand( void);
void    qh_srand( int seed);


/* ------------------------------ globals -------------------*/

/* No state is carried between rbox requests */
typedef struct rboxT rboxT;
struct rboxT {
  FILE *fout;
  FILE *ferr;
  int isinteger;
  double out_offset;
  jmp_buf errexit;        /* exit label for rboxpoints, defined by setjmp(), called by qh_errexit_rbox() */
};


int rbox_inuse= 0;
rboxT rbox;

/*-<a                             href="qh-qhull.htm#TOC"
  >-------------------------------</a><a name="rboxpoints">-</a>

  qh_rboxpoints( fout, ferr, rbox_command )
    Generate points to fout according to rbox options
    Report errors on ferr

  returns:
    0 (qh_ERRnone) on success
    1 (qh_ERRinput) on input error
    4 (qh_ERRmem) on memory error
    5 (qh_ERRqhull) on internal error

  notes:
    To avoid stdio, redefine qh_malloc, qh_free, and qh_fprintf_rbox (user.c)
    Rbox is not multithreaded.

  design:
    Straight line code (consider defining a struct and functions):

    Parse arguments into variables
    Determine the number of points
    Generate the points
*/
int qh_rboxpoints(FILE* fout, FILE* ferr, char* rbox_command) {
  int i,j,k;
  int gendim;
  int cubesize, diamondsize, seed=0, count, apex;
  int dim=3 , numpoints= 0, totpoints, addpoints=0;
  int issphere=0, isaxis=0,  iscdd= 0, islens= 0, isregular=0, iswidth=0, addcube=0;
  int isgap=0, isspiral=0, NOcommand= 0, adddiamond=0;
  int israndom=0, istime=0;
  int isbox=0, issimplex=0, issimplex2=0, ismesh=0;
  double width=0.0, gap=0.0, radius= 0.0;
  double coord[MAXdim], offset, meshm=3.0, meshn=4.0, meshr=5.0;
  double *simplex= NULL, *simplexp;
  int nthroot, mult[MAXdim];
  double norm, factor, randr, rangap, lensangle= 0, lensbase= 1;
  double anglediff, angle, x, y, cube= 0.0, diamond= 0.0;
  double box= qh_DEFAULTbox; /* scale all numbers before output */
  double randmax= qh_RANDOMmax;
  char command[200], seedbuf[200];
  char *s= command, *t, *first_point= NULL;
  time_t timedata;
  int exitcode;

  if (rbox_inuse) {
    qh_fprintf_rbox(rbox.ferr, 6188, "rbox error: rbox in use by another process.  Please lock calls to rbox.\n");
    return qh_ERRqhull;
  }
  rbox_inuse= True;
  rbox.ferr= ferr;
  rbox.fout= fout;

  exitcode= setjmp(rbox.errexit);
  if (exitcode) {
    /* same code for error exit and normal return */
    if (simplex)
        qh_free(simplex);
    rbox_inuse= False;
    return exitcode;
  }

  *command= '\0';
  strncat(command, rbox_command, sizeof(command)-strlen(command)-1);

  while (*s && !isspace(*s))  /* skip program name */
    s++;
  while (*s) {
    while (*s && isspace(*s))
      s++;
    if (*s == '-')
      s++;
    if (!*s)
      break;
    if (isdigit(*s)) {
      numpoints= qh_strtol(s, &s);
      continue;
    }
    /* ============= read flags =============== */
    switch (*s++) {
    case 'c':
      addcube= 1;
      t= s;
      while (isspace(*t))
        t++;
      if (*t == 'G')
        cube= qh_strtod(++t, &s);
      break;
    case 'd':
      adddiamond= 1;
      t= s;
      while (isspace(*t))
        t++;
      if (*t == 'G')
        diamond= qh_strtod(++t, &s);
      break;
    case 'h':
      iscdd= 1;
      break;
    case 'l':
      isspiral= 1;
      break;
    case 'n':
      NOcommand= 1;
      break;
    case 'r':
      isregular= 1;
      break;
    case 's':
      issphere= 1;
      break;
    case 't':
      istime= 1;
      if (isdigit(*s)) {
        seed= qh_strtol(s, &s);
        israndom= 0;
      }else
        israndom= 1;
      break;
    case 'x':
      issimplex= 1;
      break;
    case 'y':
      issimplex2= 1;
      break;
    case 'z':
      rbox.isinteger= 1;
      break;
    case 'B':
      box= qh_strtod(s, &s);
      isbox= 1;
      break;
    case 'D':
      dim= qh_strtol(s, &s);
      if (dim < 1
      || dim > MAXdim) {
        qh_fprintf_rbox(rbox.ferr, 6189, "rbox error: dimension, D%d, out of bounds (>=%d or <=0)", dim, MAXdim);
        qh_errexit_rbox(qh_ERRinput);
      }
      break;
    case 'G':
      if (isdigit(*s))
        gap= qh_strtod(s, &s);
      else
        gap= 0.5;
      isgap= 1;
      break;
    case 'L':
      if (isdigit(*s))
        radius= qh_strtod(s, &s);
      else
        radius= 10;
      islens= 1;
      break;
    case 'M':
      ismesh= 1;
      if (*s)
        meshn= qh_strtod(s, &s);
      if (*s == ',') {
        ++s;
        meshm= qh_strtod(s, &s);
      }else
        meshm= 0.0;
      if (*s == ',') {
        ++s;
        meshr= qh_strtod(s, &s);
      }else
        meshr= sqrt(meshn*meshn + meshm*meshm);
      if (*s && !isspace(*s)) {
        qh_fprintf_rbox(rbox.ferr, 7069, "rbox warning: assuming 'M3,4,5' since mesh args are not integers or reals\n");
        meshn= 3.0, meshm=4.0, meshr=5.0;
      }
      break;
    case 'O':
      rbox.out_offset= qh_strtod(s, &s);
      break;
    case 'P':
      if (!first_point)
        first_point= s-1;
      addpoints++;
      while (*s && !isspace(*s))   /* read points later */
        s++;
      break;
    case 'W':
      width= qh_strtod(s, &s);
      iswidth= 1;
      break;
    case 'Z':
      if (isdigit(*s))
        radius= qh_strtod(s, &s);
      else
        radius= 1.0;
      isaxis= 1;
      break;
    default:
      qh_fprintf_rbox(rbox.ferr, 7070, "rbox error: unknown flag at %s.\nExecute 'rbox' without arguments for documentation.\n", s);
      qh_errexit_rbox(qh_ERRinput);
    }
    if (*s && !isspace(*s)) {
      qh_fprintf_rbox(rbox.ferr, 7071, "rbox error: missing space between flags at %s.\n", s);
      qh_errexit_rbox(qh_ERRinput);
    }
  }

  /* ============= defaults, constants, and sizes =============== */
  if (rbox.isinteger && !isbox)
    box= qh_DEFAULTzbox;
  if (addcube) {
    cubesize= (int)floor(ldexp(1.0,dim)+0.5);
    if (cube == 0.0)
      cube= box;
  }else
    cubesize= 0;
  if (adddiamond) {
    diamondsize= 2*dim;
    if (diamond == 0.0)
      diamond= box;
  }else
    diamondsize= 0;
  if (islens) {
    if (isaxis) {
        qh_fprintf_rbox(rbox.ferr, 6190, "rbox error: can not combine 'Ln' with 'Zn'\n");
        qh_errexit_rbox(qh_ERRinput);
    }
    if (radius <= 1.0) {
        qh_fprintf_rbox(rbox.ferr, 6191, "rbox error: lens radius %.2g should be greater than 1.0\n",
               radius);
        qh_errexit_rbox(qh_ERRinput);
    }
    lensangle= asin(1.0/radius);
    lensbase= radius * cos(lensangle);
  }

  if (!numpoints) {
    if (issimplex2)
        ; /* ok */
    else if (isregular + issimplex + islens + issphere + isaxis + isspiral + iswidth + ismesh) {
        qh_fprintf_rbox(rbox.ferr, 6192, "rbox error: missing count\n");
        qh_errexit_rbox(qh_ERRinput);
    }else if (adddiamond + addcube + addpoints)
        ; /* ok */
    else {
        numpoints= 50;  /* ./rbox D4 is the test case */
        issphere= 1;
    }
  }
  if ((issimplex + islens + isspiral + ismesh > 1)
  || (issimplex + issphere + isspiral + ismesh > 1)) {
    qh_fprintf_rbox(rbox.ferr, 6193, "rbox error: can only specify one of 'l', 's', 'x', 'Ln', or 'Mn,m,r' ('Ln s' is ok).\n");
    qh_errexit_rbox(qh_ERRinput);
  }

  /* ============= print header with total points =============== */
  if (issimplex || ismesh)
    totpoints= numpoints;
  else if (issimplex2)
    totpoints= numpoints+dim+1;
  else if (isregular) {
    totpoints= numpoints;
    if (dim == 2) {
        if (islens)
          totpoints += numpoints - 2;
    }else if (dim == 3) {
        if (islens)
          totpoints += 2 * numpoints;
      else if (isgap)
        totpoints += 1 + numpoints;
      else
        totpoints += 2;
    }
  }else
    totpoints= numpoints + isaxis;
  totpoints += cubesize + diamondsize + addpoints;

  /* ============= seed randoms =============== */
  if (istime == 0) {
    for (s=command; *s; s++) {
      if (issimplex2 && *s == 'y') /* make 'y' same seed as 'x' */
        i= 'x';
      else
        i= *s;
      seed= 11*seed + i;
    }
  }else if (israndom) {
    seed= (int)time(&timedata);
    sprintf(seedbuf, " t%d", seed);  /* appends an extra t, not worth removing */
    strncat(command, seedbuf, sizeof(command)-strlen(command)-1);
    t= strstr(command, " t ");
    if (t)
      strcpy(t+1, t+3); /* remove " t " */
  } /* else, seed explicitly set to n */
  qh_RANDOMseed_(seed);

  /* ============= print header =============== */

  if (iscdd)
      qh_fprintf_rbox(rbox.fout, 9391, "%s\nbegin\n        %d %d %s\n",
      NOcommand ? "" : command,
      totpoints, dim+1,
      rbox.isinteger ? "integer" : "real");
  else if (NOcommand)
      qh_fprintf_rbox(rbox.fout, 9392, "%d\n%d\n", dim, totpoints);
  else
      qh_fprintf_rbox(rbox.fout, 9393, "%d %s\n%d\n", dim, command, totpoints);

  /* ============= explicit points =============== */
  if ((s= first_point)) {
    while (s && *s) { /* 'P' */
      count= 0;
      if (iscdd)
        out1( 1.0);
      while (*++s) {
        out1( qh_strtod(s, &s));
        count++;
        if (isspace(*s) || !*s)
          break;
        if (*s != ',') {
          qh_fprintf_rbox(rbox.ferr, 6194, "rbox error: missing comma after coordinate in %s\n\n", s);
          qh_errexit_rbox(qh_ERRinput);
        }
      }
      if (count < dim) {
        for (k=dim-count; k--; )
          out1( 0.0);
      }else if (count > dim) {
        qh_fprintf_rbox(rbox.ferr, 6195, "rbox error: %d coordinates instead of %d coordinates in %s\n\n",
                  count, dim, s);
        qh_errexit_rbox(qh_ERRinput);
      }
      qh_fprintf_rbox(rbox.fout, 9394, "\n");
      while ((s= strchr(s, 'P'))) {
        if (isspace(s[-1]))
          break;
      }
    }
  }

  /* ============= simplex distribution =============== */
  if (issimplex+issimplex2) {
    if (!(simplex= (double*)qh_malloc( dim * (dim+1) * sizeof(double)))) {
      qh_fprintf_rbox(rbox.ferr, 6196, "rbox error: insufficient memory for simplex\n");
      qh_errexit_rbox(qh_ERRmem); /* qh_ERRmem */
    }
    simplexp= simplex;
    if (isregular) {
      for (i=0; i<dim; i++) {
        for (k=0; k<dim; k++)
          *(simplexp++)= i==k ? 1.0 : 0.0;
      }
      for (k=0; k<dim; k++)
        *(simplexp++)= -1.0;
    }else {
      for (i=0; i<dim+1; i++) {
        for (k=0; k<dim; k++) {
          randr= qh_RANDOMint;
          *(simplexp++)= 2.0 * randr/randmax - 1.0;
        }
      }
    }
    if (issimplex2) {
        simplexp= simplex;
      for (i=0; i<dim+1; i++) {
        if (iscdd)
          out1( 1.0);
        for (k=0; k<dim; k++)
          out1( *(simplexp++) * box);
        qh_fprintf_rbox(rbox.fout, 9395, "\n");
      }
    }
    for (j=0; j<numpoints; j++) {
      if (iswidth)
        apex= qh_RANDOMint % (dim+1);
      else
        apex= -1;
      for (k=0; k<dim; k++)
        coord[k]= 0.0;
      norm= 0.0;
      for (i=0; i<dim+1; i++) {
        randr= qh_RANDOMint;
        factor= randr/randmax;
        if (i == apex)
          factor *= width;
        norm += factor;
        for (k=0; k<dim; k++) {
          simplexp= simplex + i*dim + k;
          coord[k] += factor * (*simplexp);
        }
      }
      for (k=0; k<dim; k++)
        coord[k] /= norm;
      if (iscdd)
        out1( 1.0);
      for (k=0; k < dim; k++)
        out1( coord[k] * box);
      qh_fprintf_rbox(rbox.fout, 9396, "\n");
    }
    isregular= 0; /* continue with isbox */
    numpoints= 0;
  }

  /* ============= mesh distribution =============== */
  if (ismesh) {
    nthroot= (int)(pow((double)numpoints, 1.0/dim) + 0.99999);
    for (k=dim; k--; )
      mult[k]= 0;
    for (i=0; i < numpoints; i++) {
      for (k=0; k < dim; k++) {
        if (k == 0)
          out1( mult[0] * meshn + mult[1] * (-meshm));
        else if (k == 1)
          out1( mult[0] * meshm + mult[1] * meshn);
        else
          out1( mult[k] * meshr );
      }
      qh_fprintf_rbox(rbox.fout, 9397, "\n");
      for (k=0; k < dim; k++) {
        if (++mult[k] < nthroot)
          break;
        mult[k]= 0;
      }
    }
  }
  /* ============= regular points for 's' =============== */
  else if (isregular && !islens) {
    if (dim != 2 && dim != 3) {
      qh_fprintf_rbox(rbox.ferr, 6197, "rbox error: regular points can be used only in 2-d and 3-d\n\n");
      qh_errexit_rbox(qh_ERRinput);
    }
    if (!isaxis || radius == 0.0) {
      isaxis= 1;
      radius= 1.0;
    }
    if (dim == 3) {
      if (iscdd)
        out1( 1.0);
      out3n( 0.0, 0.0, -box);
      if (!isgap) {
        if (iscdd)
          out1( 1.0);
        out3n( 0.0, 0.0, box);
      }
    }
    angle= 0.0;
    anglediff= 2.0 * PI/numpoints;
    for (i=0; i < numpoints; i++) {
      angle += anglediff;
      x= radius * cos(angle);
      y= radius * sin(angle);
      if (dim == 2) {
        if (iscdd)
          out1( 1.0);
        out2n( x*box, y*box);
      }else {
        norm= sqrt(1.0 + x*x + y*y);
        if (iscdd)
          out1( 1.0);
        out3n( box*x/norm, box*y/norm, box/norm);
        if (isgap) {
          x *= 1-gap;
          y *= 1-gap;
          norm= sqrt(1.0 + x*x + y*y);
          if (iscdd)
            out1( 1.0);
          out3n( box*x/norm, box*y/norm, box/norm);
        }
      }
    }
  }
  /* ============= regular points for 'r Ln D2' =============== */
  else if (isregular && islens && dim == 2) {
    double cos_0;

    angle= lensangle;
    anglediff= 2 * lensangle/(numpoints - 1);
    cos_0= cos(lensangle);
    for (i=0; i < numpoints; i++, angle -= anglediff) {
      x= radius * sin(angle);
      y= radius * (cos(angle) - cos_0);
      if (iscdd)
        out1( 1.0);
      out2n( x*box, y*box);
      if (i != 0 && i != numpoints - 1) {
        if (iscdd)
          out1( 1.0);
        out2n( x*box, -y*box);
      }
    }
  }
  /* ============= regular points for 'r Ln D3' =============== */
  else if (isregular && islens && dim != 2) {
    if (dim != 3) {
      qh_fprintf_rbox(rbox.ferr, 6198, "rbox error: regular points can be used only in 2-d and 3-d\n\n");
      qh_errexit_rbox(qh_ERRinput);
    }
    angle= 0.0;
    anglediff= 2* PI/numpoints;
    if (!isgap) {
      isgap= 1;
      gap= 0.5;
    }
    offset= sqrt(radius * radius - (1-gap)*(1-gap)) - lensbase;
    for (i=0; i < numpoints; i++, angle += anglediff) {
      x= cos(angle);
      y= sin(angle);
      if (iscdd)
        out1( 1.0);
      out3n( box*x, box*y, 0.0);
      x *= 1-gap;
      y *= 1-gap;
      if (iscdd)
        out1( 1.0);
      out3n( box*x, box*y, box * offset);
      if (iscdd)
        out1( 1.0);
      out3n( box*x, box*y, -box * offset);
    }
  }
  /* ============= apex of 'Zn' distribution + gendim =============== */
  else {
    if (isaxis) {
      gendim= dim-1;
      if (iscdd)
        out1( 1.0);
      for (j=0; j < gendim; j++)
        out1( 0.0);
      out1( -box);
      qh_fprintf_rbox(rbox.fout, 9398, "\n");
    }else if (islens)
      gendim= dim-1;
    else
      gendim= dim;
    /* ============= generate random point in unit cube =============== */
    for (i=0; i < numpoints; i++) {
      norm= 0.0;
      for (j=0; j < gendim; j++) {
        randr= qh_RANDOMint;
        coord[j]= 2.0 * randr/randmax - 1.0;
        norm += coord[j] * coord[j];
      }
      norm= sqrt(norm);
      /* ============= dim-1 point of 'Zn' distribution ========== */
      if (isaxis) {
        if (!isgap) {
          isgap= 1;
          gap= 1.0;
        }
        randr= qh_RANDOMint;
        rangap= 1.0 - gap * randr/randmax;
        factor= radius * rangap / norm;
        for (j=0; j<gendim; j++)
          coord[j]= factor * coord[j];
      /* ============= dim-1 point of 'Ln s' distribution =========== */
      }else if (islens && issphere) {
        if (!isgap) {
          isgap= 1;
          gap= 1.0;
        }
        randr= qh_RANDOMint;
        rangap= 1.0 - gap * randr/randmax;
        factor= rangap / norm;
        for (j=0; j<gendim; j++)
          coord[j]= factor * coord[j];
      /* ============= dim-1 point of 'Ln' distribution ========== */
      }else if (islens && !issphere) {
        if (!isgap) {
          isgap= 1;
          gap= 1.0;
        }
        j= qh_RANDOMint % gendim;
        if (coord[j] < 0)
          coord[j]= -1.0 - coord[j] * gap;
        else
          coord[j]= 1.0 - coord[j] * gap;
      /* ============= point of 'l' distribution =============== */
      }else if (isspiral) {
        if (dim != 3) {
          qh_fprintf_rbox(rbox.ferr, 6199, "rbox error: spiral distribution is available only in 3d\n\n");
          longjmp(rbox.errexit,qh_ERRinput);
        }
        coord[0]= cos(2*PI*i/(numpoints - 1));
        coord[1]= sin(2*PI*i/(numpoints - 1));
        coord[2]= 2.0*(double)i/(double)(numpoints-1) - 1.0;
      /* ============= point of 's' distribution =============== */
      }else if (issphere) {
        factor= 1.0/norm;
        if (iswidth) {
          randr= qh_RANDOMint;
          factor *= 1.0 - width * randr/randmax;
        }
        for (j=0; j<dim; j++)
          coord[j]= factor * coord[j];
      }
      /* ============= project 'Zn s' point in to sphere =============== */
      if (isaxis && issphere) {
        coord[dim-1]= 1.0;
        norm= 1.0;
        for (j=0; j<gendim; j++)
          norm += coord[j] * coord[j];
        norm= sqrt(norm);
        for (j=0; j<dim; j++)
          coord[j]= coord[j] / norm;
        if (iswidth) {
          randr= qh_RANDOMint;
          coord[dim-1] *= 1 - width * randr/randmax;
        }
      /* ============= project 'Zn' point onto cube =============== */
      }else if (isaxis && !issphere) {  /* not very interesting */
        randr= qh_RANDOMint;
        coord[dim-1]= 2.0 * randr/randmax - 1.0;
      /* ============= project 'Ln' point out to sphere =============== */
      }else if (islens) {
        coord[dim-1]= lensbase;
        for (j=0, norm= 0; j<dim; j++)
          norm += coord[j] * coord[j];
        norm= sqrt(norm);
        for (j=0; j<dim; j++)
          coord[j]= coord[j] * radius/ norm;
        coord[dim-1] -= lensbase;
        if (iswidth) {
          randr= qh_RANDOMint;
          coord[dim-1] *= 1 - width * randr/randmax;
        }
        if (qh_RANDOMint > randmax/2)
          coord[dim-1]= -coord[dim-1];
      /* ============= project 'Wn' point toward boundary =============== */
      }else if (iswidth && !issphere) {
        j= qh_RANDOMint % gendim;
        if (coord[j] < 0)
          coord[j]= -1.0 - coord[j] * width;
        else
          coord[j]= 1.0 - coord[j] * width;
      }
      /* ============= write point =============== */
      if (iscdd)
        out1( 1.0);
      for (k=0; k < dim; k++)
        out1( coord[k] * box);
      qh_fprintf_rbox(rbox.fout, 9399, "\n");
    }
  }

  /* ============= write cube vertices =============== */
  if (addcube) {
    for (j=0; j<cubesize; j++) {
      if (iscdd)
        out1( 1.0);
      for (k=dim-1; k>=0; k--) {
        if (j & ( 1 << k))
          out1( cube);
        else
          out1( -cube);
      }
      qh_fprintf_rbox(rbox.fout, 9400, "\n");
    }
  }

  /* ============= write diamond vertices =============== */
  if (adddiamond) {
    for (j=0; j<diamondsize; j++) {
      if (iscdd)
        out1( 1.0);
      for (k=dim-1; k>=0; k--) {
        if (j/2 != k)
          out1( 0.0);
        else if (j & 0x1)
          out1( diamond);
        else
          out1( -diamond);
      }
      qh_fprintf_rbox(rbox.fout, 9401, "\n");
    }
  }

  if (iscdd)
    qh_fprintf_rbox(rbox.fout, 9402, "end\nhull\n");

  /* same code for error exit and normal return */
  if (simplex)
    qh_free(simplex);
  rbox_inuse= False;
  return qh_ERRnone;
} /* rboxpoints */

/*------------------------------------------------
outxxx - output functions
*/
int roundi( double a) {
  if (a < 0.0) {
    if (a - 0.5 < INT_MIN) {
      qh_fprintf_rbox(rbox.ferr, 6200, "rbox input error: negative coordinate %2.2g is too large.  Reduce 'Bn'\n", a);
      qh_errexit_rbox(qh_ERRinput);
    }
    return (int)(a - 0.5);
  }else {
    if (a + 0.5 > INT_MAX) {
      qh_fprintf_rbox(rbox.ferr, 6201, "rbox input error: coordinate %2.2g is too large.  Reduce 'Bn'\n", a);
      qh_errexit_rbox(qh_ERRinput);
    }
    return (int)(a + 0.5);
  }
} /* roundi */

void out1(double a) {

  if (rbox.isinteger)
    qh_fprintf_rbox(rbox.fout, 9403, "%d ", roundi( a+rbox.out_offset));
  else
    qh_fprintf_rbox(rbox.fout, 9404, qh_REAL_1, a+rbox.out_offset);
} /* out1 */

void out2n( double a, double b) {

  if (rbox.isinteger)
    qh_fprintf_rbox(rbox.fout, 9405, "%d %d\n", roundi(a+rbox.out_offset), roundi(b+rbox.out_offset));
  else
    qh_fprintf_rbox(rbox.fout, 9406, qh_REAL_2n, a+rbox.out_offset, b+rbox.out_offset);
} /* out2n */

void out3n( double a, double b, double c) {

  if (rbox.isinteger)
    qh_fprintf_rbox(rbox.fout, 9407, "%d %d %d\n", roundi(a+rbox.out_offset), roundi(b+rbox.out_offset), roundi(c+rbox.out_offset));
  else
    qh_fprintf_rbox(rbox.fout, 9408, qh_REAL_3n, a+rbox.out_offset, b+rbox.out_offset, c+rbox.out_offset);
} /* out3n */

void qh_errexit_rbox(int exitcode)
{
    longjmp(rbox.errexit, exitcode);
} /* rbox_errexit */
