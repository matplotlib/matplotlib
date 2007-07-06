/*
** ~ppr/src/include/interface.h
** Copyright 1995, Trinity College Computing Center.
** Written by David Chappell.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** These values are used in communication between the
** interface and pprdrv and between pprdrv and pprd.
**
** Don't confuse this file with interfaces.h.
**
** This file defines the exit codes for an interface as well as the
** codes it can expect as its fifth parameter.
**
** There is no reason for the user to change anything in this file.
**
** This file was last modified 20 October 1995.
*/

/* exit levels for interfaces and pprdrv */
#define EXIT_PRINTED 0          /* file was printed normally */
#define EXIT_PRNERR 1           /* printer error occured */
#define EXIT_PRNERR_NORETRY 2   /* printer error with no hope of retry */
#define EXIT_JOBERR 3           /* job is defective */
#define EXIT_SIGNAL 4           /* terminated after catching signal */
#define EXIT_ENGAGED 5		/* printer is otherwise engaged */
#define EXIT_STARVED 6		/* starved for system resources */
#define EXIT_INTMAX EXIT_STARVED/* highest code an interface should use */

#define EXIT_INCAPABLE 7        /* printer wants features or resources */
#define EXIT_MAX EXIT_INCAPABLE /* last valid exit code */

/* the possible jobbreak methods */
#define JOBBREAK_DEFAULT -1	/* <-- not a real setting, used only in ppad */
#define JOBBREAK_NONE 0
#define JOBBREAK_SIGNAL 1
#define JOBBREAK_CONTROL_D 2
#define JOBBREAK_PJL 3
#define JOBBREAK_SIGNAL_PJL 4
#define JOBBREAK_SAVE_RESTORE 5
#define JOBBREAK_NEWINTERFACE 6

/* end of file */
