/* Very simple interface to the ppr TT routines */
/* (c) Frank Siegert 1996 */

#include <stdio.h>
#include <stdarg.h>

void fatal(int rval, const char *string, ...)
    {
    va_list va;

    va_start(va,string);
                             
    fprintf(stderr,"FATAL %d: ",rval);
    vfprintf(stderr,string,va);
    fprintf(stderr,"\n");
    
    /* We probably don't have to do this since we */
    /* are exiting, but then, maybe we do. */
    va_end(va);

    exit(rval);         /* die now */     
}

void debug(const char *string, ... )
    {
    va_list va;
    FILE *file;

    va_start(va,string);
    fprintf( stderr, "DEBUG:");
    vfprintf(stderr,string,va);
    fprintf(stderr,"\n");
    va_end(va);
}


void *myalloc(int size, int length)
{
	char *ret=(char *)malloc(size*(length+1));
	return (void *)ret;
}

int myfree(void *toBeFreed)
{
	free(toBeFreed);
	return 0;
}

void *myrealloc(void *original, int newsize)
{
	return (void *)realloc(original, newsize);
}


printer_putc(int val)
{
	putc(val,stdout);
}

printer_putline(char *a)
{
	printf("%s\n",a);
}

int main(int argc, char **argv) {
	if (argc!=2) {
		fprintf(stderr,"ttconv V1.1\nwritten by Frank Siegert out of PPR\nThis programs converts a ttfont to PS Type 3, it needs a file name as input parameter, all output goes to stdout, debugging to stderr\n");
		exit(0);
	}
	insert_ttfont(argv[1]);
	return 0;
}
