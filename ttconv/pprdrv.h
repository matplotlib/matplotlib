/*
** ~ppr/src/include/pprdrv.h
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
** This file last revised 5 December 1995.
*/

/* 
** No debug code will be included if this
** is not defined: 
*/
/* #define DEBUG 1 */

/*
** Uncomment the defines for the debugging
** code you want to have included.
*/
#ifdef DEBUG
#define DEBUG_TRUETYPE 		/* truetype fonts, conversion to Postscript */
/* #define DEBUG_PROGINIT 1 */		/* main() */
/* #define DEBUG_PPD 1 */		/* PPD file parsing */
/* #define DEBUG_INTERFACE 1 */		/* show opening, closing, etc. */
/* #define DEBUG_FLAGS 1 */		/* banner and trailer pages */
/* #define DEBUG_RESOURCES 1 */		/* fonts, procset, etc. */
/* #define DEBUG_QUERY 1 */		/* patch queries? */
#define DEBUG_TWOWAY 1			/* show twoway operations */
#define DEBUG_TWOWAY_LINES 1		/* show the lines read from printer */
/* #define DEBUG_BINSELECT_INLINE 1 */	/* add comments to output */
#endif

#define PPRDRV_LOGFILE LOGDIR"/pprdrv"

#define PPD_TABSIZE 128		/* initial size of PPD file string hash table */
#define FONT_TABSIZE 64		/* initial size of font hash table */
#define MAX_PPDNAME 50		/* max length of PPD string name */
#define MAX_PPDTEXT 4096	/* max length of PPD string value */
#define MAX_DRVRES 200		/* maximum resource in document */
#define MAX_DRVREQ 40		/* maximum requirements in a document */
#define MAX_PAPERSIZES 20	/* maximum *PaperDimension lines */

/*
** What should be done if we have trouble
** converting a TrueType font to PostScript?
*/
#define EXIT_TTFONT EXIT_PRNERR_NORETRY

/* GDBM file name for table of TrueType fonts. */
#define TTDBNAME CONFDIR"/ttfonts"

/* Do not change anything below this line. */

/* pprdrv.c */
extern char line[];
extern int line_len;
extern int line_overflow;
char *dgetline(FILE *infile);
char *dgetline_read(FILE *infile);
extern char *QueueFile;
extern struct QFileEntry job;
extern int group_pass;
extern int group_job;
extern int doing_primary_job;
extern int sheetcount;
extern int print_direction;
void fatal(int exval, const char *str,...);
void debug(const char *str,...);              
void error(const char *string, ... );
extern char *drvreq[MAX_DRVREQ];
extern int drvreq_count;
extern int strip_binselects;	/* for pprdrv_ppd.c */
extern int strip_signature;	/* for pprdrv_ppd.c */
struct DRVRES *add_drvres(int needed, int fixinclude, char *type, char *name,
	double version, int revision);
int add_resource(char *type, char *name, double version, int revision);
void add_resource_font(char *name);
void jobbreak(void);

/* pprdrv_flag.c */
void print_flag_page(int flagtype, int possition);
char *reverse_for(void);

/* pprdrv_twoway.c */
void ready_twoway(void);
void start_twoway(void);
void close_log(void);
extern int posterror;		/* TRUE if error message rec'v'd */
extern int ghostscript;		/* TRUE if we get Ghostscript style error messages */
void twoway_pjl_wait(void);

/* routines in pprdrv_ppd.c */
void add_font(char *fontname);
void new_string(char *name);
void string_line(char *string);
void end_string(void);
void read_PPD_file(char *name);
char *find_feature(char *name, char *variation);
void begin_stopped(void);
void end_stopped(char *feature, char *option);
void insert_features(FILE *qstream, int set);    
void include_feature(char *featuretype, char *option);
void begin_feature(char *featuretype, char *option, FILE *infile);
void _include_resource(char *type, char *name, double version, int revision);
void include_resource(void);
int find_font(char *fontname);                  
void papersize_moveto(char *paper);

/* Routines in pprdrv_res.c */
void insert_noinclude_fonts(void);
void insert_extra_prolog_resources(void);
void write_resource_comments(void);
void begin_resource(FILE *infile);

/* pprdrv_capable.c */
int check_if_capable(FILE *qfile);      

/* pprdrv_media.c */
int load_mountedlist(void);
int select_media(char *name);
int select_media_by_dsc_name(char *name);
void read_media_lines(FILE *q);
extern struct Media_Xlate media_xlate[];
extern int media_count;

/* routines in pprdrv_buf.c */
void printer_puts_QuotedValue(char *str);
void printer_puts_escaped(char *str);
void printer_putc(int c);
void printer_puts(char *str);
void printer_putline(char *str);
void printer_printf(char *str, ...);
void printer_write(char *buf, int len);
void printer_flush(void);
void bufinit(void);

/* routines in pprdrv_nup.c */
void prestart_N_Up_hook(void);
void invoke_N_Up(void);
void close_N_Up(void);

/* routines in pprdrv_req.c */
void write_requirement_comments(void);

/* routines in pprdrv_signature.c */
int signature(int sheetnumber,int thissheet);

/* routines in pprdrv_reason.c */
void give_reason(const char *reason);
void describe_postscript_error(const char *error, const char *command);

/* routines in pprdrv_patch.c */
void patchfile(void);
void jobpatchfile(void);
int patchfile_query_callback(char *message);

/* routines in pprdrv_commentary.c */
void commentary(int flags, char *message);

/* routines in pprdrv_tt.c */
char *find_ttfont(char *name);
void want_ttrasterizer(void);
void insert_ttfont(char *filename);

/* routines in pprdrv_progress.c */
void progress__page_start_comment_sent(void);
void progress__pages_truly_printed(int n);
void progress__bytes_sent(int n);

/*
** A record which describes a printer commentator which
** should be fed information about what is going on with
** the printer.
*/
struct COMMENTATOR
	{
	char *progname;		/* "file" or program to invoke */
	char *address;		/* first parameter to feed to it */
	int interests;		/* bitmask telling when to invoke */
	struct COMMENTATOR *next;
	} ;

/*
** The values for the "flags" argument to the commentary() 
** function.  These are used to indicate into which 
** catagory the message falls so that commentary() can decide
** which of the commentators to pass the information to.
*/
#define COM_PRINTER_ERRORS 1	/* printer errors ( %%[ PrinterError: xxxxxx ]%% ) */
#define COM_PRINTER_STATUS 2	/* printer status messages ( %%[ status: xxxxxx ]%% ) */
#define COM_IMPATIENCE 4	/* "printing bogged down" */
#define COM_EXIT 8		/* Why did it exit? */

/*
** the information we compile about the printer and 
** the current job
*/
struct PPRDRV {
        char *Name;                         
        char *Interface;                    /* stuff */
        char *Address;                      /* from */
        char *Options;                      /* printer */
	struct COMMENTATOR *Commentators;   /* the list of processes to tell about things */
        int charge;                         /* page charge in cents */
        int do_banner;                      /* configuration */
        int do_trailer;                     /* file */
        int feedback;                       /* true or false */
        int jobbreak;                       /* one of several values */
        int OutputOrder;                    /* 1 or -1 or 0 if unknown */
        char *PPDFile;                      /* name of description file */
	int type42_ok;			    /* Can we use type42 fonts? */
	int GrayOK;			    /* permission to print non-colour jobs? */
        } ;

/* This structure is in pprdrv.c */
struct PPRDRV printer;

/*
** A PPD file string entry.
*/
struct PPDSTR {
        char *name;
        char *value;
        struct PPDSTR *next;
        } ;

/*
** A PPD file font entry.
*/
struct PPDFONT {
        char *name;
        struct PPDFONT *next;
        } ;

/*
** The list of device features 
*/
struct FEATURES {
    int ColorDevice;                /* TRUE or FALSE, real colour printing */
    int Extensions;                 /* 0 or EXTENSION_* */
    int FaxSupport;                 /* FAXSUPPORT_* */
    int FileSystem;                 /* TRUE or FALSE */
    int LanguageLevel;              /* 1, 2, etc. */
    int TTRasterizer;		    /* TT_NONE, TT_ACCEPT68K, TT_TYPE42 */
    } ;

/* This structure is in pprdrv_ppd.c */
struct FEATURES Features;

/*
** Structure to describe a mounted media list entry.
** The mounted media list files are generated by pprd,
** but pprd uses two fwrite calles for each record and does
** not use this structure.
*/
struct MOUNTED
    {
    char bin[MAX_BINNAME];
    char media[MAX_MEDIANAME];
    } ;

/* this structure is in pprdrv_media.c */
extern struct MOUNTED mounted[MAX_BINS];

/*
** Structure used by pprdrv to describe a resource:
*/
struct DRVRES
    {
    char *type;             /* procset, font, etc. */
    char *name;             /* name of this resource */
    double version;         /* procset only */
    int revision;           /* procset only */
    int needed;             /* TRUE or FALSE */
    int fixinclude;         /* TRUE or FALSE */
    int force_into_docsetup;/* TRUE or FALSE */
    char *former_name;	    /* name of substituted resource */
    char *subcode;	    /* PostScript code for font substitution */
    char *filename;	    /* NULL or file to load cached resource from */
    int truetype;	    /* Resource is a TrueType font to be converted to type 42 */
    int mactt;		    /* Resource is a TrueType font converted by a Mac to PostScript */
    int force_into_prolog;  /* Insert resource in prolog */
    } ;

extern struct DRVRES drvres[];  /* in pprdrv.c */
extern int drvres_count;        /* in pprdrv.c */

/*
** Structure in pprdrv_ppd.c for paper sizes.
*/
struct PAPERSIZE
    {
    char *name;			/* name of paper size */
    double width;		/* width in 1/72ths */
    double height;
    double lm;			/* left margin */
    double tm;			/* top margin */
    double rm;			/* right margin */
    double bm;			/* bottom margin */
    } ;

extern struct PAPERSIZE papersize[];
extern int papersizex;
extern int num_papersizes;

/*
** This structure contains information we will use
** to monitor how well writes are progressing.
*/
struct WRITEMON
    {
    int interval;
    int impatience;
    } ;
    
extern struct WRITEMON writemon;    

/*
** The structure which is used to translate between PPR media names and 
** the media names in the "%%Media:" lines.
*/
struct 	Media_Xlate {
	char *pprname;
	char *dscname;
	} ;

/* end of file */
