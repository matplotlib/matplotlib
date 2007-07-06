
CFLAGS=-g -DDEBUG_TRUETYPE

PROGS = ttconv
OBJS  = pprdrv_tt.o pprdrv_tt2.o ttmain.o

all: ${PROGS}

ttconv: pprdrv_tt.o pprdrv_tt2.o ttmain.o
	${CC} ${CFLAGS} pprdrv_tt.o pprdrv_tt2.o ttmain.o -o ttconv

${OBJS}:	truetype.h interface.h pprdrv.h global_defines.h 

clean:
	-rm -f ${OBJS} a.out core

clobber:	clean
	-rm -f ${PROGS}

.c.o: 
	${CC} ${CFLAGS} -c $*.c


