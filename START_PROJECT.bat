@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title MachineGuard AI - CNC Maintenance

echo =========================================================
echo   MachineGuard AI - Windows One-Click Starter
 echo =========================================================
echo.

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python 3.10 to 3.13 is required.
        echo Download Python from https://www.python.org/downloads/
        echo Tick "Add Python to PATH" during installation.
        goto :failed
    )
    set "PYTHON=python"
)

%PYTHON% -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), 'Install Python 3.10 to 3.13'"
if errorlevel 1 (
    echo [ERROR] Unsupported Python version. Install Python 3.10 to 3.13.
    goto :failed
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Preparing isolated Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :failed
)

if not exist ".venv\.deps_installed" (
    echo [2/4] Installing required Python libraries - internet needed on first launch...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :failed
    echo Installed> ".venv\.deps_installed"
) else (
    echo [2/4] Using installed dependencies.
)

echo [3/4] Preparing dataset and trained model...
".venv\Scripts\python.exe" bootstrap.py --prefer-official
if errorlevel 1 goto :failed

echo [4/4] Opening browser app...
".venv\Scripts\python.exe" launcher.py
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo [ERROR] MachineGuard could not start. Read the message above.
echo See README.md under Troubleshooting for solutions.
pause
exit /b 1
