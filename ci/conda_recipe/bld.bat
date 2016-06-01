set LIBPATH=%LIBRARY_LIB%;
set INCLUDE=%INCLUDE%;%PREFIX%\Library\include\freetype2

ECHO [directories] > setup.cfg
ECHO basedirlist = %LIBRARY_PREFIX% >> setup.cfg
ECHO [packages] >> setup.cfg
ECHO tests = False >> setup.cfg
ECHO sample_data = False >> setup.cfg
ECHO toolkits_tests = False >> setup.cfg

@rem workaround for https://github.com/matplotlib/matplotlib/issues/6460
@rem see also https://github.com/conda-forge/libpng-feedstock/pull/4
copy /y %LIBRARY_LIB%\libpng16.lib %LIBRARY_LIB%\png.lib

%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
