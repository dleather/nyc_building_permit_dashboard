@echo off
REM This batch file runs CodeWeaver in the current directory
REM and excludes the specified directories and files from the documentation.

REM Define ignore list patterns (each pattern on its own line for clarity)
set "IGNORE_LIST=\.git.*"
set "IGNORE_LIST=%IGNORE_LIST%,__pycache__"
set "IGNORE_LIST=%IGNORE_LIST%,\.venv"
set "IGNORE_LIST=%IGNORE_LIST%,node_modules"
set "IGNORE_LIST=%IGNORE_LIST%,build"
set "IGNORE_LIST=%IGNORE_LIST%,.*\.log"
set "IGNORE_LIST=%IGNORE_LIST%,temp"
set "IGNORE_LIST=%IGNORE_LIST%,^data(/|\\)"
set "IGNORE_LIST=%IGNORE_LIST%,README.html"
set "IGNORE_LIST=%IGNORE_LIST%,README.md"
set "IGNORE_LIST=%IGNORE_LIST%,README_files"
set "IGNORE_LIST=%IGNORE_LIST%,uv.lock"
set "IGNORE_LIST=%IGNORE_LIST%,run_codeweaver.bat"
set "IGNORE_LIST=%IGNORE_LIST%,.env"

REM Run CodeWeaver using the ignore list
C:\Users\davle\go\bin\CodeWeaver.exe -input . -ignore="%IGNORE_LIST%"

REM Pause so you can see any output in the command window before it closes.
pause