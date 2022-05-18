#!/bin/bash

# Subsetting DejaVu fonts to create a display-math-only font

# The DejaVu fonts include math display variants outside of the Unicode range,
# and it is currently hard to access them from matplotlib. The subset.py script
# in `tools` has been modified to move the math display variants found in DejaVu
# fonts into a new TTF font with these variants in the Unicode range.

# This bash script calls the subset.py scripts with the appropriate options to
# generate the new font files `DejaVuSansDisplay.ttf` and
# `DejaVuSerifDisplay.ttf`:

mpldir=$(dirname $0)/../

# test that fontforge is installed
python -c 'import fontforge' 2> /dev/null
if [ $? != 0 ]; then
    echo "The python installation at $(which python) does not have fontforge"
    echo "installed. Please install it before using subset.py."
    exit 1
fi

FONTDIR=$mpldir/lib/matplotlib/mpl-data/fonts/ttf/

python $mpldir/tools/subset.py --move-display --subset=dejavu-ext $FONTDIR/DejaVuSans.ttf \
    $FONTDIR/DejaVuSansDisplay.ttf
python $mpldir/tools/subset.py --move-display --subset=dejavu-ext $FONTDIR/DejaVuSerif.ttf \
    $FONTDIR/DejaVuSerifDisplay.ttf
