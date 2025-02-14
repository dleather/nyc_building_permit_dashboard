@echo off
REM This batch file runs CodeWeaver in the current directory
REM and excludes the specified directories and files from the documentation.

C:\Users\davle\go\bin\CodeWeaver.exe -input . -ignore="\.git.*,__pycache__,\.venv,node_modules,build,.*\.log,temp,data"

REM Pause so you can see any output in the command window before it closes.
pause