@echo off
title FloraVision AI - Flower Species Classification 
color 0A

echo ======================================================================
echo           FLORAVISION AI - FLOWER SPECIES CLASSIFIER 
echo ======================================================================
echo.

:: Detect Python
set PYTHON_CMD=python
where py >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.13
)

echo [1/2] Checking Python environment...
%PYTHON_CMD% --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo.
echo [2/2] Launching FloraVision AI Web Server...
echo ----------------------------------------------------------------------
echo   Local PC:    http://localhost:5000
echo   Phone / LAN: Check the IP shown below (Phone must be on same Wi-Fi)
echo ----------------------------------------------------------------------
echo.

:: Open default browser after 2 seconds in background
start "" cmd /c "timeout /t 2 >nul & start http://localhost:5000"

:: Start Flask app
%PYTHON_CMD% app.py

pause
