@echo off
title XAUUSD Backtest Lab
cd /d "%~dp0"

set "PY="
py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY py -3.13 -c "import sys" >nul 2>&1 && set "PY=py -3.13"
if not defined PY py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY python -c "import sys" >nul 2>&1 && set "PY=python"
if not defined PY goto nopython

if exist ".venv\Scripts\python.exe" goto haveenv
echo First run: creating the Python environment...
%PY% -m venv .venv
if errorlevel 1 goto failed
:haveenv

if exist ".venv\installed.ok" goto run
echo Installing packages - one time only, takes 2 to 4 minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo ok> ".venv\installed.ok"

:run
echo.
echo  XAUUSD Backtest Lab is starting. Your browser opens at http://localhost:8501
echo  Keep this window open while you use the app. Close it to stop.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
exit /b 0

:nopython
echo Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/
pause
exit /b 1

:failed
echo Setup failed. Delete the .venv folder and run this file again.
pause
exit /b 1
