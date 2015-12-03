:: To build extensions for 64 bit Python 3, we need to configure environment
:: variables to use the MSVC 2010 C++ compilers from GRMSDKX_EN_DVD.iso of:
:: MS Windows SDK for Windows 7 and .NET Framework 4 (SDK v7.1)
::
:: To build extensions for 64 bit Python 2, we need to configure environment
:: variables to use the MSVC 2008 C++ compilers from GRMSDKX_EN_DVD.iso of:
:: MS Windows SDK for Windows 7 and .NET Framework 3.5 (SDK v7.0)
::
:: 32 bit builds do not require specific environment configurations.
::
:: Note: this script needs to be run with the /E:ON and /V:ON flags for the
:: cmd interpreter, at least for (SDK v7.0)
::
:: More details at:
:: https://github.com/cython/cython/wiki/64BitCythonExtensionsOnWindows
:: http://stackoverflow.com/a/13751649/163740
::
:: Author: Olivier Grisel
:: License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/
::@ECHO OFF

SET COMMAND_TO_RUN=%*
SET WIN_SDK_ROOT=C:\Program Files\Microsoft SDKs\Windows

:: unquote
call :unquote PYTHON_VERSION %PYTHON_VERSION%
SET MAJOR_PYTHON_VERSION=%PYTHON_VERSION:~0,1%
IF "%PYTHON_VERSION:~1,1%" == "." (
    :: CONDA_PY style, such as 27, 34 etc.
    SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~1,1%
) ELSE (
    IF "%PYTHON_VERSION:~3,1%" == "." (
     SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,1%
    ) ELSE (
     SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,2%
    )
)
SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,1%
set USE_MS_SDK=N
IF %MAJOR_PYTHON_VERSION% == 2 (
    SET WINDOWS_SDK_VERSION=v7.0
    set USE_MS_SDK=Y
) ELSE IF %MAJOR_PYTHON_VERSION% == 3 (
    rem py3.5 does not need a sdk set...
    IF %MINOR_PYTHON_VERSION% LEQ 4 (
        SET WINDOWS_SDK_VERSION=v7.1
        set USE_MS_SDK=Y
    )   
) ELSE (
    ECHO Unsupported Python version: "%MAJOR_PYTHON_VERSION%"
    EXIT 1
)

SET WINDOWS_SDK_VERSION=%WINDOWS_SDK_VERSION%
IF "%PYTHON_ARCH%"=="64" (
    IF "%USE_MS_SDK%" == "N" (
        echo Using the happy new world of py35+ auto configuring compilers....
    ) ELSE (
        ECHO Configuring Windows SDK %WINDOWS_SDK_VERSION% for Python %MAJOR_PYTHON_VERSION%.%MINOR_PYTHON_VERSION% on a 64 bit architecture
        SET DISTUTILS_USE_SDK=1
        SET MSSdk=1
        "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Setup\WindowsSdkVer.exe" -q -version:%WINDOWS_SDK_VERSION%
        "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Bin\SetEnv.cmd" /x64 /release
    )
) ELSE (
    ECHO Using default MSVC build environment for 32 bit architecture
)
ECHO Executing: %COMMAND_TO_RUN%
call %COMMAND_TO_RUN% || EXIT 1
goto :EOF

:unquote
  set %1=%~2
  goto :EOF
