#!/bin/bash

if [ `uname` == Linux ]; then
    pushd $PREFIX/lib
    ln -s libtcl8.5.so libtcl.so
    ln -s libtk8.5.so libtk.so
    popd
fi

cat <<EOF > setup.cfg
[directories]
basedirlist = $PREFIX

[packages]
tests = False
toolkit_tests = False
sample_data = False

EOF

# The macosx backend isn't building with conda at this stage.
if [ `uname` == Darwin ]; then
cat << EOF >> setup.cfg

[gui_support]
tkagg = true
macosx = false

EOF
fi

cat setup.cfg
sed -i.bak "s|/usr/local|$PREFIX|" setupext.py


$PYTHON setup.py install

