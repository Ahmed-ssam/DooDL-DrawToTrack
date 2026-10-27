@echo off
rem One-click start for Windows: sets itself up the first time, then opens DooDL in your browser.
cd /d "%~dp0"
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY (
  echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run this file again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Setting up DooDL for the first time. This takes a minute or two...
  %PY% -m venv .venv || goto :fail
)
if not exist ".venv\.installed" (
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip || goto :fail
  ".venv\Scripts\python.exe" -m pip install --quiet -e . || goto :fail
  echo done> ".venv\.installed"
)
".venv\Scripts\python.exe" -m doodl.cli serve --open
pause
exit /b 0
:fail
echo.
echo Setup failed. Copy the messages above and send them to the project owner.
pause
exit /b 1
