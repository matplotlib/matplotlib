@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=python -msphinx
)
set SOURCEDIR=.
set BUILDDIR=build
set SPHINXPROJ=matplotlib
if defined SPHINXOPTS goto skipopts
set SPHINXOPTS=-W --keep-going
:skipopts

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The Sphinx module was not found. Make sure you have Sphinx installed,
	echo.then set the SPHINXBUILD environment variable to point to the full
	echo.path of the 'sphinx-build' executable. Alternatively you may add the
	echo.Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "html-noplot" goto html-noplot
if "%1" == "html-skip-subdirs" goto html-skip-subdirs
if "%1" == "show" goto show

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
if "%1" == "clean" (
	REM workaround because sphinx does not completely clean up (#11139)
	rmdir /s /q "%SOURCEDIR%\build"
	rmdir /s /q "%SOURCEDIR%\api\_as_gen"
	rmdir /s /q "%SOURCEDIR%\gallery"
	rmdir /s /q "%SOURCEDIR%\plot_types"
	rmdir /s /q "%SOURCEDIR%\tutorials"
	rmdir /s /q "%SOURCEDIR%\users\explain"
	rmdir /s /q "%SOURCEDIR%\savefig"
	rmdir /s /q "%SOURCEDIR%\sphinxext\__pycache__"
	del /q "%SOURCEDIR%\_static\constrained_layout*.png"
	del /q "%SOURCEDIR%\sg_execution_times.rst"
)
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html-noplot
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O% -D plot_gallery=0
goto end

:html-skip-subdirs
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O% -D skip_sub_dirs=1
goto end


:show
python -m webbrowser -t "%~dp0\build\html\index.html"

:end
popd
