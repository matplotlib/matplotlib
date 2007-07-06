/*
** ~ppr/src/include/global_defines.h
** Copyright 1995, Trinity College Computing Center.
** Written by David Chappell.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software and documentation are provided "as is" without
** express or implied warranty.
**
** The PPR project was begun 28 December 1992.
**
** There are many things in this file you may want to change.  This file 
** should be the first include file.  It is the header file for the whole 
** project.
**
** This file was last modified 22 December 1995.
*/

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdarg.h>

#define mode_t int
/* 
** Define if we should include code to make GNU-C happy.
** This generaly means initializing a variable to zero
** in order to supress incorrect warnings about possibly
** uninitialized variables.
*/
#define GNUC_HAPPY 1

/*
** Define if we want the spooler to keep superseeded
** PostScript code as comments.  These comments can be
** useful in debuging.
*/
#define KEEP_OLD_CODE 1

/*
** Define various file names.
** Those not begining with "/" origionate in HOMEDIR.
**
** Some of the names are defined only if they have not been defined
** already.  This is because one can make limited changes to PPR's
** directory layout by editing the top of globak.mk.  If you
** do, the C compiler will be invoked with command line switches
** which override the definitions here.
*/
#ifndef HOMEDIR
#define HOMEDIR "/usr/ppr"			/* directory for most permanent files */
#endif
#ifndef CONFDIR
#define CONFDIR "/etc/ppr"			/* top directory for all configuration files (must be absolute) */
#endif
#ifndef VAR_SPOOL_PPR
#define VAR_SPOOL_PPR "/var/spool/ppr"		/* are for fixed path things */
#endif
#ifndef TEMPDIR
#define TEMPDIR "/tmp"				/* for ordinary temporary files */
#endif

#define NEXTIDFILE VAR_SPOOL_PPR"/nextid"	/* file with next queue id number */
#define QUEUEDIR VAR_SPOOL_PPR"/queue"		/* queue directory */
#define DATADIR VAR_SPOOL_PPR"/jobs"		/* date directory */
#define FIFO_NAME VAR_SPOOL_PPR"/PIPE"		/* name of pipe between ppr & pprd */
#define LOCKFILE VAR_SPOOL_PPR"/pprd.lock"      /* created and locked by pprd */
#define ALERTDIR VAR_SPOOL_PPR"/alerts"		/* directory for alert files */
#define LOGDIR VAR_SPOOL_PPR"/logs"		/* directory for log files */
#define INTDIR "interfaces"			/* directory for interface programs */
#define COMDIR "commentators"			/* directory for commentator programs */
#define PRE_CACHEDIR "cache"			/* pre-loaded cache files */
#define CACHEDIR VAR_SPOOL_PPR"/cache"		/* directory for cache files */
#define RESPONDERDIR "responders"		/* responder programs */
#define PPDDIR "PPDFiles"			/* our PPD file library */
#define MOUNTEDDIR CONFDIR"/mounted"		/* directory for media mounted files */
#define PRCONF CONFDIR"/printers"		/* printer configuration files */
#define GRCONF CONFDIR"/groups"			/* group configuration files */
#define DBNAME CONFDIR"/charge_users"		/* users database file name */
#define DEFFILTOPTS CONFDIR"/deffiltopts"	/* directory for default filter options */
#define GROUPSPPD CONFDIR"/groups.ppd"		/* directory for auto-generated group PPD files */
#define FONTSUB CONFDIR"/fontsub"		/* font substitution database */
#define MEDIAFILE CONFDIR"/media"		/* media definitions */
#define PRINTLOG "printlog"			/* needed so that pprd won't delete */
#define PRINTLOG_PATH LOGDIR"/"PRINTLOG		/* log of jobs printed */

/* The paths of various programs which must be invoked. */
#define PPRDRV_PATH "lib/pprdrv"
#define PPOP_PATH "bin/ppop"
#define PPR_PATH "bin/ppr"

/*
** These are rather system dependent, so sysdep.h
** may redefine many of these.  We define them here
** rather than waiting to defined them conditionally
** after including sysdep.h for philisophical reasons.
** We want to make it plain that these values will
** apply to the vast majority of systems.
*/
#define SHORT_INT short int		/* a 16 bit number */
#define MAILPATH "/usr/lib/sendmail"	/* mail program for alerting */
#define SHORT_PATH "/bin:/usr/bin"	/* Secure path */

/*
** Include system dependent modifications to the stuff above
** and special defines necessary to compile on particular systems.
*/

/*
** some practical limits
*/
#define MAX_LINE 1024 /* 8192 */    /* maximum PostScript input line length (now pretty meaningless) */
#define MAX_CONT 32		    /* maximum segment represented by "%%+" */
#define MAX_TOKENIZED 512           /* longest line we may pass to tokenize() */
#define MAX_PATH 128                /* space to reserve for building a file name */
#define MAX_TOKENS 20               /* limit on words per comment line */

#define MAX_BINNAME 16              /* max chars in name of input bin */
#define MAX_MEDIANAME 16            /* max chars in media name */
#define MAX_COLOURNAME 16           /* max chars in colour name */
#define MAX_TYPENAME 16             /* max chars media type name */

#define MAX_DOCMEDIA 4              /* max media types per job */

#define QUEUE_SIZE 2000             /* 2000 entry queue */
#define MAX_DESTNAME 16             /* max length of destination name */
#define MAX_PRINTERS 150            /* no more than 150 printers */
#define MAX_BINS 4                  /* max bins per printer */
#define MAX_GROUPS 150              /* no more than this may groups */
#define MAX_GROUPSIZE 8             /* no more than 8 printers per group */

#define MAX_QFLINE 150		    /* Max length of queue file line (exclusive of newline and NULL) */
#define MAX_RESPONSE_METHOD 16	    /* Max length of method name (responder name) */
#define MAX_RESPONSE_ADDRESS 256    /* Max length of address to pass to responder */

#define MAX_CONFLINE 255	    /* Max length of config file line (exclusive of newline and NULL) */

/*
** True/False values
*/
#define APPLE_QUOTE 1               /* allow non-standard quote mark quoting */

/*=========================================================================*/
/* End of values you might want to change.                                 */
/*=========================================================================*/

/*---------------------------------------------------
** external functions in libppr.a
---------------------------------------------------*/
void *myalloc(size_t number, size_t size);
void *myrealloc(void *ptr, size_t size);
char *mystrdup(const char *string);
void myfree(void *ptr);
char *datestamp(void);
void tokenize(void);
extern char *tokens[];
void ASCIIZ_to_padded(char *padded, const char *asciiz, int len);
void padded_to_ASCIIZ(char *asciiz, const char *padded, int len);
int padded_cmp(const char *padded1, const char *padded2, int len);
int ppr_sscanf(const char *string, const char *pattern, ...);
void daemon(void);
void valert(const char printername[], int dateflag, const char string[], va_list args);
void alert(const char printername[], int dateflag, const char string[], ...);
char *quote(const char *);
double getdouble(const char *);
char *dtostr(double);
int torf(const char *s);
int destination_protected(const char *destname);
int icmp(const char *s1, const char *s2);
int icmpn(const char *s1, const char *s2, int n);
int lock_exclusive(int filenum, int waitmode);
char *money(int amount_times_ten);
char *jobid(const char *destname, int qid, int subid);
int pagesize(const char *keyword, int *width, int *length, int *envelope);
int disk_space(const char *path, int *free_blocks, int *free_files);
char *noalloc_find_cached_resource(const char *res_type, const char *res_name, double version, int revision, int *new_revision, mode_t *mode);
char *find_cached_resource(const char *res_type, const char *res_name, double version, int revision, int *new_revision, mode_t *mode);
void wrap_string(char *target, const char *source, int width);
int get_responder_width(const char *name);
void options_start(const char *options_str);
int options_get_one(char *name, int maxnamelen, char *value, int maxvaluelen);
extern const char *options_string;
extern const char *options_error;
extern int options_error_context_index;
extern int options_error_index;
double convert_dimension(const char *string);
void filter_option_error(int exlevel, const char *format, ...) __attribute__ ((noreturn));
const char *pap_strerror(int err);
const char *nbp_strerror(int err);
const char *pap_look_string(int n);

/* NEEDS_STRSIGNAL may be defined in sysdep.h */
#ifdef NEEDS_STRSIGNAL
const char *strsignal(int signum);
#endif

/*
** Functions and constant which library callers must provide if they
** call certain library functions.
*/
void fatal(int exitval, const char *string, ...) __attribute__ ((noreturn));
void error(const char *string, ...);
extern const int memory_exit;

/*
** Characters which are not allowed in printer
** and group names.
*/
#define DEST_DISALLOWED "/~.-"

/*
** TRUE and FALSE
**  The code makes liberal use of these macros.
*/
#if !defined(FALSE)
#define FALSE 0
#endif
#if !defined(TRUE)
#define TRUE !FALSE
#endif

/*
** Define unix permission 755.  We do this because just saying
** 0755 is at least theoretically non-portable and because
** this portable expression is long and unsightly.
*/
#define UNIX_755 (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH)

/*
** Values for torf(), a function which examines a string
** and tries to determine whether it represents a true or
** a false value.
*/
#define ANSWER_UNKNOWN -1           
#define ANSWER_FALSE 0
#define ANSWER_TRUE 1

/*
** Types of PostScript langauge extensions:
*/
#define EXTENSION_DPS 1
#define EXTENSION_CMYK 2
#define EXTENSION_Composite 4
#define EXTENSION_FileSystem 8

/*
** Types of fax support:
*/
#define FAXSUPPORT_NONE 0
#define FAXSUPPORT_Base 1

/*
** "%%PageOrder:" settings
*/
#define PAGEORDER_ASCEND 1
#define PAGEORDER_DESCEND -1
#define PAGEORDER_SPECIAL 0

/*
** Valid banner and trailer options.
*/
#define BANNER_DONTCARE 0           /* ppr submits the job with one */    
#define BANNER_YESPLEASE 1          /* of these */   
#define BANNER_NOTHANKYOU 2          

#define BANNER_FORBIDDEN 0          /* the printer configuration includes one */      
#define BANNER_DISCOURAGED 1        /* of these */
#define BANNER_ENCOURAGED 2
#define BANNER_REQUIRED 3   
#define BANNER_INVALID 4	    /* used to indicate invalid user input in ppad(8) */

/*
** Job status values.
** A positive value is the ID of a printer which
** is currently printing the job.
*/
#define STATUS_WAITING -1           /* waiting for printer */
#define STATUS_HELD -2              /* put on hold by user */      
#define STATUS_WAITING4MEDIA -3     /* proper media not mounted */
#define STATUS_ARRESTED -4          /* auto put on hold because of job error */
#define STATUS_CANCEL -5            /* being canceled */

/*
** Printer status values.
*/
#define PRNSTATUS_IDLE 0            /* idle but ready to print */
#define PRNSTATUS_PRINTING 1        /* printing right now */
#define PRNSTATUS_CANCELING 2       /* canceling a job */
#define PRNSTATUS_FAULT 3           /* waiting for auto retry */           
#define PRNSTATUS_ENGAGED 4	    /* printer is printing for another computer */
#define PRNSTATUS_STARVED 5	    /* starved for system resources */
#define PRNSTATUS_STOPT 6           /* stopt by user */
#define PRNSTATUS_STOPPING 7        /* will go to PRNSTATUS_STOPT at job end */
#define PRNSTATUS_HALTING 8         /* pprdrv being killed */
#define PRNSTATUS_DELETED 9	    /* printer has been deleted */
#define PRNSTATUS_DELIBERATELY_DOWN 6 /* 1st non-printing value (stopt) */

/*
** "%%ProofMode:" values.
*/
#define PROOFMODE_NOTIFYME -1
#define PROOFMODE_SUBSTITUTE 0		/* default mode */
#define PROOFMODE_TRUSTME 1

/*
** Signiture part values.
*/
#define SIG_FRONTS 1
#define SIG_BACKS -1
#define SIG_BOTH 0

/*
** Flags stored in the unix file permissions of a font:
*/
#define FONT_MACTRUETYPE S_IXUSR	/* Is a Macintosh TrueType font in PostScript form */
#define FONT_TYPE1 S_IXGRP		/* Type 1 components present */
#define FONT_TYPE42 S_IXOTH		/* Type 42 components present */

/*
** Valid TrueType rasterizer settings.
*/
#define TT_UNKNOWN 0
#define TT_NONE 1
#define TT_ACCEPT68K 2
#define TT_TYPE42 3


/* end of file */
