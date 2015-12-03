mkdir lib
copy %LIBRARY_LIB%\zlibstatic.lib lib\z.lib
if errorlevel 1 exit 1
copy %LIBRARY_LIB%\libpng_static.lib lib\png.lib
if errorlevel 1 exit 1

set MPLBASEDIRLIST=%LIBRARY_PREFIX%;.

:: debug...
set

copy setup.cfg.template setup.cfg
if errorlevel 1 exit 1

python setup.py install
if errorlevel 1 exit 1

rd /s /q %SP_DIR%\dateutil
rd /s /q %SP_DIR%\numpy

if "%ARCH%"=="64" (
    set PLAT=win-amd64
) else (
    set PLAT=win32
)

::copy C:\Tcl%ARCH%\bin\t*.dll %SP_DIR%\matplotlib-%PKG_VERSION%-py%PY_VER%-%PLAT%.egg\matplotlib\backends
