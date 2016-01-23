set LIB=%LIBRARY_LIB%
set LIBPATH=%LIBRARY_LIB%;
set INCLUDE=%LIBRARY_INC%;%PREFIX%\Library\include\freetype2

ECHO [directories] > setup.cfg
ECHO basedirlist = %LIBRARY_PREFIX% >> setup.cfg
ECHO [packages] >> setup.cfg
ECHO tests = False >> setup.cfg
ECHO sample_data = False >> setup.cfg
ECHO toolkits_tests = False >> setup.cfg

python setup.py install
if errorlevel 1 exit 1
