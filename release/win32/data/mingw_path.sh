#!/bin/sh

# This script will convert a UNIX path to a Win32 native path
UPATH=$1
if test "$UPATH" = ""; then
    echo EMPTY
    exit 1
fi
#echo "INPUT IS \"$UPATH\"" >&2
if [ -d "$UPATH" ]
then
    cd "$UPATH"
    WPATH=`pwd -W`
else
    # cd up to parent directories until we find
    # one that exists. Loop ends at "/".
    dpart=`dirname "$UPATH"`
    #echo "dpart starts as \"$dpart\"" >&2
    while [ ! -d "$dpart" ]
    do
        dpart=`dirname "$dpart"`
        #echo "dpart is \"$dpart\"" >&2
    done
    #echo "dpart ends as \"$dpart\"" >&2

    if [ "$dpart" != "." ]
    then
        dstart=`expr length "$dpart"`
        # If the last character in dpart is not "/",
        # then advance dstart by one index. This
        # avoids two dir seperators in the result.
        last=`expr length "$dpart"`
        last=`expr $last - 1`
        last=${dpart:$last:1}
        #echo "last is \"$last\"" >&2
        if [ "$last" != "/" ]
        then
            dstart=`expr $dstart + 1`
        fi
        dend=`expr length "$UPATH"`
        dlen=`expr $dend - $dstart`
        #echo "UPATH is \"$UPATH\"" >&2
        #echo "dstart is $dstart, dend is $dend, dlen is $dlen" >&2
        bpart=${UPATH:$dstart:$dend}
        dpart=`cd "$dpart" ; pwd -W`
        #echo "dpart \"$dpart\"" >&2
        #echo "bpart \"$bpart\"" >&2
    else
        dpart=`pwd -W`
        bpart=$UPATH
    fi
    WPATH=${dpart}/${bpart}
fi
#echo "OUTPUT IS \"$WPATH\"" >&2
echo $WPATH
exit 0

