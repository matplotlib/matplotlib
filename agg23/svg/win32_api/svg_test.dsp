# Microsoft Developer Studio Project File - Name="svg_test" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Application" 0x0101

CFG=svg_test - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "svg_test.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "svg_test.mak" CFG="svg_test - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "svg_test - Win32 Release" (based on "Win32 (x86) Application")
!MESSAGE "svg_test - Win32 Debug" (based on "Win32 (x86) Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "svg_test - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /I "../../include" /I "../../../Expat/Source/lib" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /FD /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x419 /d "NDEBUG"
# ADD RSC /l 0x419 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib libexpat.lib /nologo /subsystem:windows /profile /machine:I386 /out:"./svg_test.exe" /libpath:"../../../Expat/libs"

!ELSEIF  "$(CFG)" == "svg_test - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /I "../../include" /I "../../../Expat/Source/lib" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /YX /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x419 /d "_DEBUG"
# ADD RSC /l 0x419 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib libexpat.lib /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept /libpath:"../../../Expat/libs"

!ENDIF 

# Begin Target

# Name "svg_test - Win32 Release"
# Name "svg_test - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\src\agg_bezier_arc.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_curves.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_gsv_text.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_path_storage.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\platform\win32\agg_platform_support.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_rasterizer_scanline_aa.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\ctrl\agg_slider_ctrl.cpp
# End Source File
# Begin Source File

SOURCE=..\agg_svg_parser.cpp
# End Source File
# Begin Source File

SOURCE=..\agg_svg_path_renderer.cpp
# End Source File
# Begin Source File

SOURCE=..\agg_svg_path_tokenizer.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_trans_affine.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_vcgen_contour.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_vcgen_dash.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_vcgen_markers_term.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_vcgen_stroke.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\agg_vpgen_clip_polygon.cpp
# End Source File
# Begin Source File

SOURCE=..\..\src\platform\win32\agg_win32_bmp.cpp
# End Source File
# Begin Source File

SOURCE=..\svg_test.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
